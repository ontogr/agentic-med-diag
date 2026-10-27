// Two-edge paths from seed entities in both directions, capped per seed.
// $ids: seed Entity id strings.
// $per_seed: maximum paths returned for each seed.
// $exclude: relation types to skip, such as hierarchy edges.
UNWIND $ids AS eid
CALL {
  WITH eid
  MATCH (a:Entity {id: eid})-[r1:RELATES_TO]-(m:Entity)-[r2:RELATES_TO]-(c:Entity)
  WHERE a <> c AND NOT r1.type IN $exclude AND NOT r2.type IN $exclude
  // FILTERS
  RETURN a, r1, m, r2, c
  ORDER BY coalesce(r1.score, 0) + coalesce(r2.score, 0) DESC, r1.id, r2.id
  LIMIT $per_seed
}
RETURN a.id AS head_id, a.name AS head_name, r1.type AS first_type,
       m.id AS via_id, m.name AS via_name, r2.type AS second_type,
       c.id AS tail_id, c.name AS tail_name
