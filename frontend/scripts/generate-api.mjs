// Схема строится тем же FastAPI factory без запуска lifespan, БД или LLM.
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { writeFile, mkdir } from 'node:fs/promises'
import openapiTS, { astToString } from 'openapi-typescript'

const root = fileURLToPath(new URL('../../', import.meta.url))
execFileSync(
  process.env.PYTHON || 'python',
  [
    '-m',
    'agent_app.service.openapi_cli',
    '--config',
    'config/web_contract.yaml',
    '--output',
    'data/openapi/support-api.json',
  ],
  { cwd: root, stdio: 'inherit' },
)
const ast = await openapiTS(new URL('../../data/openapi/support-api.json', import.meta.url))
const target = new URL('../src/shared/api/schema.d.ts', import.meta.url)
await mkdir(new URL('../src/shared/api/', import.meta.url), { recursive: true })
await writeFile(
  target,
  '// Сгенерировано из OpenAPI; изменения выполняются через npm run api:generate.\n' +
    astToString(ast),
)
