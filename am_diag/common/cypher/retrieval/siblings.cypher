// Entities of the same label that share a neighbour with a seed entity.
// $ids: seed Entity id strings.
// $max_siblings: maximum entities returned.
UNWIND $ids AS eid
MATCH (e:Entity {id: eid})-[:RELATES_TO]-(p:Entity)-[:RELATES_TO]-(s:Entity)
WHERE NOT s.id IN $ids
  AND any(l IN labels(s) WHERE l <> 'Entity' AND l IN labels(e))
// FILTERS
RETURN DISTINCT s.id AS id, s.name AS name, labels(s) AS labels
LIMIT $max_siblings
