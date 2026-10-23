MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
RETURN source.id AS source, target.id AS target, coalesce(r.score, 1.0) AS score
ORDER BY source, target, r.id
SKIP $offset
LIMIT $limit
