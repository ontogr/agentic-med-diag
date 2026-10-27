// Source document title and external id for each chunk.
// $ids: Chunk id strings.
UNWIND $ids AS cid
MATCH (c:Chunk {id: cid})-[:PART_OF]->(d:Document)
RETURN c.id AS chunk_id, d.title AS title, d.external_id AS external_id
