// Chunks that mention a neighbour of a seed entity.
// $ids: seed Entity id strings.
// $limit: maximum chunks returned.
UNWIND $ids AS eid
MATCH (e:Entity {id: eid})-[:RELATES_TO]-(n:Entity)<-[:HAS_ENTITY]-(c:Chunk)
RETURN DISTINCT c.id AS chunk_id
LIMIT $limit
