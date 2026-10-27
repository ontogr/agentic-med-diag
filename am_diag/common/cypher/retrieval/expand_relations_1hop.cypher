// Edges around seed entities in both directions, capped per seed.
// $ids: seed Entity id strings.
// $per_seed: maximum edges returned for each seed.
// $exclude: relation types to skip, such as hierarchy edges.
UNWIND $ids AS eid
CALL {
  WITH eid
  MATCH (a:Entity {id: eid})-[r:RELATES_TO]-(b:Entity)
  WHERE NOT r.type IN $exclude
  // FILTERS
  RETURN r
  ORDER BY coalesce(r.score, 0) DESC, r.id
  LIMIT $per_seed
}
RETURN r.id AS id, r.type AS type, r.description AS description,
       r.structural_label AS structural_label, r.sources AS sources,
       r.chunk_ids AS chunk_ids, r.score AS score,
       startNode(r).id AS head_id, startNode(r).name AS head_name,
       endNode(r).id AS tail_id, endNode(r).name AS tail_name
