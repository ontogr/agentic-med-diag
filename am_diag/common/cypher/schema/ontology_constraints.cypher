// Uniqueness for the ontology mirror nodes, which use reserved __Onto labels.
// Without these, concurrent registrations could create duplicate type nodes.
CREATE CONSTRAINT onto_entity_label IF NOT EXISTS
FOR (n:`__OntoEntity`) REQUIRE n.label IS UNIQUE;
CREATE CONSTRAINT onto_meta_key IF NOT EXISTS
FOR (n:`__OntoMeta`) REQUIRE n.key IS UNIQUE;
