// Chunks that mention each seed entity.
// $ids: seed Entity id strings.
UNWIND $ids AS eid
MATCH (c:Chunk)-[:HAS_ENTITY]->(e:Entity {id: eid})
RETURN eid AS entity_id, c.id AS chunk_id
