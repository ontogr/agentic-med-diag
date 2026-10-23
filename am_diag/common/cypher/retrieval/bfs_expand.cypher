// BFS expansion from seed Entity nodes up to $depth hops via :RELATES_TO.
// $seed_ids: list of string UUIDs to start from.
// $depth: max hop count (default 2).
// $limit: max nodes to return (caps fan-out for medical supernodes).
UNWIND $seed_ids AS sid
MATCH (s:Entity {id: sid})
MATCH path = (s)-[:RELATES_TO*1..${depth}]->(n:Entity)
WITH n, min(length(path)) AS hop,
     collect(DISTINCT last(relationships(path)).type)[0..3] AS rels
WHERE hop <= $depth
RETURN n.id AS id, n.label AS label, n.name AS name, hop, rels
ORDER BY hop ASC, n.name
LIMIT $limit
