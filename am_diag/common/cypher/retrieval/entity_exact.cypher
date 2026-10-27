// Entities whose name, canonical name, alias or synonym equals a keyword.
// $keywords: keyword strings.
// $per_keyword: maximum entities returned for each keyword.
UNWIND $keywords AS kw
CALL {
  WITH kw
  MATCH (e:Entity)
  WHERE (toLower(e.name) = toLower(kw)
     OR toLower(coalesce(e.canonical_name, '')) = toLower(kw)
     OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) = toLower(kw))
     OR any(s IN coalesce(e.synonyms, []) WHERE toLower(s) = toLower(kw)))
  // FILTERS
  RETURN e
  ORDER BY e.id
  LIMIT $per_keyword
}
RETURN e.id AS id, e.name AS name, labels(e) AS labels,
       e.description AS description, e.curie AS curie
