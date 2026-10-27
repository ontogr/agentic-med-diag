// Dynamic enums populated at call time via TypeBuilder.
// The @@dynamic marker tells BAML that values come from the TypeBuilder, not
// from a static list here — callers must pass {"tb": tb} to every function
// that uses these types.
enum EntityType { @@dynamic }
enum RelationType { @@dynamic }

class ExtractedEntity {
  name string
    @description("Canonical name. Use generic/INN for drugs, HGNC symbol for genes.")
  entity_type EntityType
    @description("Must be one of the configured entity types.")
  description string?
    @description("Optional brief description of this entity from the text.")
  aliases string[]
    @description("Abbreviations or synonyms found in text.")
  properties map<string, string>
    @description("Type-specific properties: e.g. icd10Code for Disease, mechanism for Drug.")
  score float
    @description("1.0 = explicitly named, 0.7 = strongly implied, 0.5 = inferred")
}

class ExtractedRelation {
  head_name string
    @description("Canonical name of the head entity — must appear in entities list")
  tail_name string
    @description("Canonical name of the tail entity — must appear in entities list")
  relation_type RelationType
    @description("Must be one of the configured relation types.")
  score float
  properties map<string, string>
    @description("Relation-specific properties.")
}

class ExtractedKnowledgeGraph {
  entities ExtractedEntity[]
  relations ExtractedRelation[]
}

// Type enums are injected via TypeBuilder; allowed endpoint pairs go in the prompt.
// Pass {"tb": build_schema_typebuilder(schema)} at call time.

function ExtractKnowledgeGraph(
  text: string,
  source_type: string,
  relation_guide: string,
) -> ExtractedKnowledgeGraph {
  client OpenAIGeneric
  prompt #"
    {{ _.role("system") }}
    You are a biomedical knowledge graph extraction expert. Extract all medical entities and
    relationships from the text using ONLY the entity and relation types available to you.

    RULES:
    1. Only extract entities explicitly present in the text — no hallucination
    2. entity_type must be one of the valid types (constrained by the system)
    3. relation_type must be one of the valid types (constrained by the system)
    4. Both head_name and tail_name of every relation must appear in the entities list
    5. Scores: 1.0 = explicitly named, 0.7 = strongly implied, 0.5 = inferred
    6. Use properties map for type-specific fields — only include if stated in text
    7. Source context: {{ source_type }}
    8. A relation is valid only when its head and tail types match a listed pair.
       Use the listed direction. Do not output a relation when no pair fits.

    Allowed relation types and endpoint pairs:
    {{ relation_guide }}

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Extract the knowledge graph from this text:

    {{ text }}
  "#
}

// Entity coreference arbiter

class SamenessVerdict {
  areSameEntity bool
  canonicalName string?
    @description("If same, the preferred canonical name (generic INN for drugs, HGNC for genes)")
  reasoning string
    @description("One sentence: key evidence for or against")
}

function AreEntityMentionsSame(
  entityA: string,
  typeA: string,
  entityB: string,
  typeB: string,
) -> SamenessVerdict {
  client OpenAIGeneric

  prompt #"
    {{ _.role("system") }}
    You are a biomedical entity disambiguation expert.
    Determine if two mentions refer to the same real-world entity.

    Consider: abbreviations (MI=myocardial infarction), generic/brand (metformin=Glucophage),
    synonyms. Do NOT merge different subtypes (type 1 ≠ type 2 diabetes).

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Entity A: "{{ entityA }}" (type: {{ typeA }})
    Entity B: "{{ entityB }}" (type: {{ typeB }})
    Same entity?
  "#
}

// Cluster deduplication verdict (used by EntityEdgeResolver)

class ClusterDeduplicationVerdict {
  duplicates string[]
    @description("Exact matches to items in the candidate set that mean the same thing as the item")
  alias string
    @description("Best canonical name representing item + duplicates, ideally chosen from the candidate set (Wikidata-alias style)")
  reasoning string
    @description("One sentence: why these are/aren't duplicates and why this alias was chosen")
}

function IdentifyClusterDuplicates(
  item: string,
  itemKind: string,
  candidateSet: string[],
) -> ClusterDeduplicationVerdict {
  client OpenAIGeneric

  prompt #"
    {{ _.role("system") }}
    You are a biomedical knowledge graph deduplication expert.
    Given one {{ itemKind }} and a candidate set of other {{ itemKind }}s from the
    same semantic cluster, identify which candidates are EXACT duplicates of the
    item — i.e. they refer to the same real-world entity or relationship concept,
    differing only in tense, plural/singular form, stem form, casing, abbreviation,
    or shorthand. Do NOT merge items that differ in clinically meaningful ways
    (e.g. "type 1 diabetes" vs "type 2 diabetes", "causes" vs "treats").

    Then choose the single best canonical alias to represent the item and its
    duplicates together — prefer a full, unabbreviated, singular, lowercase form
    from the candidate set or the item itself (Wikidata-alias style: the most
    widely recognised name for the shared concept).

    Return an empty `duplicates` list if there are no exact duplicates; `alias`
    must always be set (to the item itself if there are no duplicates).

    {{ ctx.output_format }}

    {{ _.role("user") }}
    Item ({{ itemKind }}): "{{ item }}"

    Candidate set:
    {% for candidate in candidateSet %}
    - "{{ candidate }}"
    {% endfor %}

    Which candidates are exact duplicates of the item, and what is the best alias?
  "#
}
