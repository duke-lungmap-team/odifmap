import ontospy
import pickle
import os
from .resources import resource_path

LOCAL_ONTOLOGY_FILE = os.path.join(resource_path, 'lung_ontology.owl')
PICKLED_ONTOLOGY = os.path.join(resource_path, 'lung_ontology.pkl')

try:
    # load pickled ontospy object
    f = open(PICKLED_ONTOLOGY, 'rb')
    onto = pickle.load(f)
    f.close()
except FileNotFoundError:
    onto = ontospy.Ontospy(uri_or_path=LOCAL_ONTOLOGY_FILE, rdf_format='xml')

    # pickle the ontology
    f = open(PICKLED_ONTOLOGY, 'wb')
    pickle.dump(onto, f)
    f.close()


def get_onto_protein_uri(ontology, protein_label):
    sparql_proteins_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX : <http://www.semanticweb.org/am175/ontologies/2017/1/untitled-ontology-79#>
SELECT ?p ?p_label WHERE {
    ?p rdfs:subClassOf :Protein .
    ?p :has_synonym ?p_label .
    VALUES ?p_label { "%s" }
}
""" % protein_label

    results = ontology.query(sparql_proteins_query)

    return results


def get_onto_cells_by_protein(ontology, protein_uri):
    sparql_protein_cell_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX : <http://www.semanticweb.org/am175/ontologies/2017/1/untitled-ontology-79#>
SELECT ?c WHERE {
    ?c rdfs:subClassOf* :cell . 
    ?c rdfs:subClassOf ?restriction .
    ?restriction owl:onProperty :has_part ; owl:someValuesFrom ?p .
    VALUES ?p { <%s> }
}
""" % protein_uri

    results = ontology.query(sparql_protein_cell_query)

    return results


def get_onto_tissues_by_cell(ontology, cell_uri):
    sparql_cell_tissue_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX : <http://www.semanticweb.org/am175/ontologies/2017/1/untitled-ontology-79#>
SELECT ?t WHERE {
    ?t rdfs:subClassOf* :tissue .
    ?t rdfs:subClassOf ?restriction .
    ?restriction owl:onProperty :has_part ; owl:someValuesFrom ?c .
    VALUES ?c { <%s> }
}
""" % cell_uri

    results = ontology.query(sparql_cell_tissue_query)

    return results


def get_onto_structures_by_related_uri(ontology, uri):
    sparql_tissue_structure_query1 = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX : <http://www.semanticweb.org/am175/ontologies/2017/1/untitled-ontology-79#>
SELECT ?s ?label ?pred WHERE {
    ?s rdfs:subClassOf* :complex_structure .
    ?s :lungmap_preferred_label ?label . 
    ?s rdfs:subClassOf ?restriction .
    ?restriction owl:onProperty ?pred ; owl:someValuesFrom ?t .
    VALUES ?t { <%s> } .
    VALUES ?pred { :has_part :surrounded_by }
}
""" % uri

    results = ontology.query(sparql_tissue_structure_query1)

    return results


def get_onto_sub_classes(ontology, uri):
    sparql_subclass_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX : <http://www.semanticweb.org/am175/ontologies/2017/1/untitled-ontology-79#>
SELECT ?sub ?label WHERE {
    ?sub rdfs:subClassOf ?uri . 
    ?sub :lungmap_preferred_label ?label . 
    VALUES ?uri { <%s> }
}
""" % uri

    results = ontology.query(sparql_subclass_query)

    return results
