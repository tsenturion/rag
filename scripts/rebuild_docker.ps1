#requires -Version 7.0

<#
.SYNOPSIS
Собирает и атомарно обновляет локальный Docker-контур проекта.

.DESCRIPTION
Команда сохраняет именованные volumes, но принудительно пересоздаёт API и фоновые
workers. После запуска проверяется, что все Python-сервисы используют один digest
образа rag-support-agent и не исполняют разные версии исходного кода.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$compose = @(
    "compose",
    "--profile", "observability",
    "--profile", "orchestration",
    "--profile", "bpmn"
)

function Get-DotEnvValue {
    <#
    .SYNOPSIS
    Читает одно значение из корневого .env без вывода секрета в журнал сборки.
    #>
    param([Parameter(Mandatory)][string]$Name)

    $envPath = Join-Path $PSScriptRoot "..\.env"
    if (-not (Test-Path -LiteralPath $envPath)) {
        return $null
    }
    $prefix = "$Name="
    $line = Get-Content -LiteralPath $envPath |
        Where-Object { $_.TrimStart().StartsWith($prefix) } |
        Select-Object -Last 1
    if (-not $line) {
        return $null
    }
    $value = ($line -split "=", 2)[1].Trim()
    if (
        $value.Length -ge 2 -and
        (($value.StartsWith("'") -and $value.EndsWith("'")) -or
        ($value.StartsWith('"') -and $value.EndsWith('"')))
    ) {
        return $value.Substring(1, $value.Length - 2)
    }
    return $value
}

docker @compose config --quiet
if ($LASTEXITCODE -ne 0) {
    throw "Итоговая Docker Compose-конфигурация не прошла проверку."
}

# orchestration-worker использует тот же image/tag и Dockerfile, что support-agent.
# Одна сборка исключает гонку двух BuildKit exporters при записи общего latest.
docker @compose build --pull support-agent code-runner
if ($LASTEXITCODE -ne 0) {
    throw "Не удалось собрать проектные Docker images."
}

docker @compose up -d --force-recreate --remove-orphans
if ($LASTEXITCODE -ne 0) {
    throw "Не удалось пересоздать Docker Compose-контур."
}

# Docker environment применяется к новым процессам, но не меняет учётные
# записи, сохранённые внутри именованных volumes. Синхронизация нужна после
# ротации .env, иначе workers получат новый пароль RabbitMQ, а broker оставит
# старый; Grafana аналогично продолжит принимать прежний admin password.
$rabbitUser = Get-DotEnvValue "RABBITMQ_DEFAULT_USER"
$rabbitPassword = Get-DotEnvValue "RABBITMQ_DEFAULT_PASS"
if ($rabbitUser -and $rabbitPassword) {
    docker @compose exec -T rabbitmq rabbitmqctl change_password `
        $rabbitUser $rabbitPassword | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось синхронизировать пароль RabbitMQ с .env."
    }
}

$grafanaPassword = Get-DotEnvValue "GRAFANA_ADMIN_PASSWORD"
if ($grafanaPassword) {
    $grafanaPassword | docker @compose exec -T grafana grafana cli `
        --homepath /usr/share/grafana admin reset-admin-password `
        --password-from-stdin | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось синхронизировать пароль Grafana с .env."
    }
}

$agentServices = @(
    "support-agent",
    "orchestration-worker",
    "camunda-worker",
    "flower"
)
$digests = foreach ($service in $agentServices) {
    $containerId = (docker compose ps -q $service).Trim()
    if (-not $containerId) {
        throw "После обновления не найден контейнер сервиса $service."
    }
    $imageId = (docker inspect --format '{{.Image}}' $containerId).Trim()
    [pscustomobject]@{ Service = $service; ImageId = $imageId }
}

if (($digests.ImageId | Sort-Object -Unique).Count -ne 1) {
    $details = $digests | Format-Table -AutoSize | Out-String
    throw "Python-сервисы используют разные image digest:`n$details"
}

$digests | Format-Table -AutoSize
docker @compose ps
