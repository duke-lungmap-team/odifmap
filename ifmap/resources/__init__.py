from pkg_resources import resource_stream, resource_filename
import pathlib

resource_path = resource_filename('ifmap', 'resources')
# force POSIX-style path, even on Windows
resource_path = pathlib.Path(resource_path).as_posix()

lung_ontology = resource_stream('ifmap.resources', 'lung_ontology.owl')