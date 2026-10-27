// Retrieval functions: text-to-Cypher generation and query keyword extraction

class CypherQuery {
  cypher string
    @description("A valid read-only Cypher MATCH/RETURN query answering the question. No mutations.")
  reasoning string
    @description("One sentence: what the query finds and why this structure was chosen")
}

function GenerateCypherQuery(
  question: string,
  schema_description: string,
  examples: string,
  feedback: string,
) -> CypherQuery {
  client OpenAIGeneric

  prompt #"
    {{ _.role("system") }}
    You are a Cypher query expert for a biomedical knowledge graph. The query may
    run on Neo4j or FalkorDB. Generate a read-only Cypher query that answers the
    clinical question.

    Knowledge graph schema:
    {{ schema_description }}

    Example questions and queries:

    Q: Which drugs are indicated for type 2 diabetes?
    MATCH (d:Entity:Drug)-[r:RELATES_TO {type: 'INDICATION'}]->(x:Entity:Disease)
    WHERE toLower(x.name) CONTAINS 'type 2 diabetes'
    RETURN d.name AS drug LIMIT 25

    Q: Which genes are associated with Alzheimer disease?
    MATCH (x:Entity:Disease)-[r:RELATES_TO {type: 'ASSOCIATED_WITH'}]-(g:Entity:Gene)
    WHERE toLower(x.name) CONTAINS 'alzheimer'
    RETURN g.name AS gene LIMIT 25

    Q: How many phenotypes are present in cystic fibrosis?
    MATCH (x:Entity:Disease)-[r:RELATES_TO {type: 'PHENOTYPE_PRESENT'}]->(p:Entity:Phenotype)
    WHERE toLower(x.name) CONTAINS 'cystic fibrosis'
    RETURN count(DISTINCT p) AS phenotypes

    {% if examples %}
    More example queries for reference:
    {{ examples }}
    {% endif %}

    RULES:
    1. Only use MATCH / OPTIONAL MATCH / WITH / WHERE / RETURN — no CREATE / MERGE / DELETE / SET
    2. Use ``n.id`` to identify nodes. Use ``n.name`` for human-readable names.
    3. Use specific node labels (e.g. ``Drug``, ``Disease``) when the question implies them.
    4. Every edge is ``(a:Entity)-[r:RELATES_TO {type: '...'}]->(b:Entity)``. Filter
       relations by the ``type`` property, never by a relationship type name.
    5. Include a LIMIT clause (maximum 50) to prevent runaway queries.
    6. Return ``n.name``, ``n.id``, ``r.type`` — not raw internal IDs alone.
    7. Give every returned value a column name with ``AS``.

    {% if feedback %}
    The previous query was rejected: {{ feedback }}
    Fix this problem in the new query.
    {% endif %}

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Question: {{ question }}
  "#
}

class QueryKeywords {
  entities string[]
    @description("Clinical entities named in the question: diseases, drugs, genes, phenotypes, anatomy, pathways, exposures. Include common abbreviations.")
  terms string[]
    @description("Other important content words from the question, at most 5")
}

function ExtractQueryKeywords(
  question: string,
  schema_text: string,
) -> QueryKeywords {
  client OpenAIGeneric

  prompt #"
    {{ _.role("system") }}
    Extract search keywords from a biomedical question. The knowledge graph
    holds these types:
    {{ schema_text }}

    Return entity names as a clinician would write them, and add their usual
    abbreviations. Do not return question words such as "what" or "which".

    {{ ctx.output_format }}

    {{ _.role("user") }}
    {{ question }}
  "#
}
