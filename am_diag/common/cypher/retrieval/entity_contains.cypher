// Entities whose name contains a keyword, excluding exact name matches.
// $keywords: keyword strings.
// $per_keyword: maximum entities returned for each keyword.
UNWIND $keywords AS kw
CALL {
  WITH kw
  MATCH (e:Entity)
  WHERE toLower(e.name) CONTAINS toLower(kw) AND toLower(e.name) <> toLower(kw)
  // FILTERS
  RETURN e
  ORDER BY size(e.name) ASC, toLower(e.name) ASC, e.id ASC
  LIMIT $per_keyword
}
RETURN e.id AS id, e.name AS name, labels(e) AS labels,
       e.description AS description, e.curie AS curie
