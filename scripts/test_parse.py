import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.lmstudio import parse_triples_flexible

samples = [
    '{"triples":[{"subject":"Анна","relation":"работает_в","object":"Сбер"}]}',
    'Вот ответ:\n```json\n{"triples":[{"subject":"Анна","relation":"работает_в","object":"Сбер"}]}\n```',
    '[{"субъект":"Пётр","связь":"знает","объект":"Маша"}]',
    "Анна — работает_в → Сбер\nПётр — знает → Маша",
]

for s in samples:
    ts = parse_triples_flexible(s)
    print("---")
    print([(t.subject, t.relation, t.object) for t in ts])
