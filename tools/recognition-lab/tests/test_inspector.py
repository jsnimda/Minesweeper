import numpy as np
from grid_geometry.phase.global_geometry import GlobalGeometry, GlobalResult
from grid_geometry.phase.tiles import LatticeConsensus
from grid_geometry.phase.decoder import DecodeResult

from recognition_lab.inspector import inspector_sections, property_report
from recognition_lab.recognition_state import RecognitionSelection, ResultId
from recognition_lab.results import Crop, Detection, GeometryResult, RecognitionResult


def values(sections):
    return {section.title: {prop.name: prop.value for prop in section.properties} for section in sections}


def sample(accepted=True, mode='whole', with_extent=True):
    lattice = LatticeConsensus(12.123456789012345, 13.25, 0.125, 0.0, (0,), (), False, 0.0, .125)
    extent = GlobalGeometry(np.array([1.125, 13.5, 25.75]), np.array([2.25, 15.5]), (0, 2), (0, 1)) if with_extent else None
    validation = GlobalResult(lattice, extent, accepted, () if accepted else ('termination unobservable',),
                              np.zeros((1, 1)), np.zeros((1, 1)), measurements={'coverage': 0.0, 'valid': False})
    grid = GeometryResult(0, Crop((1, 2, 31, 42), False), np.zeros((1, 1, 1)), None,
                          DecodeResult('fitted', ()), validation, [], diagnostics={'mode': mode, 'tile count': 0},
                          phase_data={'global': validation, 'observations': []})
    return RecognitionResult([Detection('grid', .9876543210987654, (1.125, 2.25, 30.5, 40.75))], [grid])


def test_detection_properties_and_precision():
    state = RecognitionSelection(); state.replace(sample())
    section = values(inspector_sections(state.snapshot, state.snapshot.roots[0].id))['Detection']
    assert section['Class'] == 'grid' and section['Class ID'] == 'Unavailable'
    assert section['Detection index (zero-based)'] == '0'
    assert section['Confidence'] == repr(state.snapshot.result.detections[0].confidence)
    assert section['Left (px)'] == '1.125' and section['Width (px)'] == '29.375'
    assert section['Coordinate space'] == 'Screenshot pixels'


def test_geometry_properties_and_missing_optional_values():
    state = RecognitionSelection(); state.replace(sample(mode='tiled'))
    node = state.snapshot.roots[0].children[0]
    sections = values(inspector_sections(state.snapshot, node.id))
    assert sections['Geometry result']['Originating detection index (zero-based)'] == '0'
    assert sections['Geometry result']['Inference mode (recorded)'] == 'tiled'
    assert sections['Geometry result']['Other run settings'] == 'Unavailable'
    assert sections['Input crop']['Clipped'] == 'No'
    assert sections['Input crop']['Width (px)'] == '30'
    assert sections['Stored outputs']['Observations'] == '0'
    assert sections['Stored outputs']['Heatmaps'].startswith('Placeholder')
    assert sections['Recorded diagnostics']['tile count'] == '0'
    grid = state.snapshot.result.grids[0]
    grid.validation = None; grid.heatmaps = None; grid.phase_data = None; grid.diagnostics = {}
    state.replace(state.snapshot.result)
    sections = values(inspector_sections(state.snapshot, state.snapshot.roots[0].children[0].id))
    assert sections['Geometry result']['Validation status'] == 'Unavailable'
    assert sections['Geometry result']['Inference mode (recorded)'] == 'Unavailable'
    assert sections['Stored outputs']['Fitted lattice'] == 'No'
    assert sections['Stored outputs']['Heatmaps'] == 'Unavailable'
    assert sections['Stored outputs']['Observations'] == 'Unavailable'


def test_lattice_coordinates_and_validation_extent():
    state = RecognitionSelection(); state.replace(sample())
    lattice_node = state.snapshot.roots[0].children[0].children[0]
    sections = values(inspector_sections(state.snapshot, lattice_node.id))
    lattice = sections['Fitted lattice']
    assert lattice['Coordinate space'] == 'Resized crop pixels (valid region, padding removed)'
    assert lattice['Origin Y (px)'] == '0.0' and lattice['Ambiguous'] == 'No'
    assert lattice['Pitch X (px)'] == repr(state.snapshot.resolve(lattice_node.id).pitch_x)
    assert lattice['Rejected observations'] == '0'
    extent = sections['Validation-derived board extent']
    assert extent['Rows'] == '1' and extent['Columns'] == '2'
    assert extent['Left edge (px)'] == '1.125'
    state.replace(sample(mode='tiled', with_extent=False))
    node = state.snapshot.roots[0].children[0].children[0]
    sections = values(inspector_sections(state.snapshot, node.id))
    assert sections['Fitted lattice']['Coordinate space'] == 'Crop pixels (native resolution)'
    assert sections['Validation-derived board extent']['Board extent'] == 'Unavailable'
    state.replace(sample(mode=None))
    node = state.snapshot.roots[0].children[0].children[0]
    assert values(inspector_sections(state.snapshot, node.id))['Fitted lattice']['Coordinate space'] == 'Unavailable (inference mode not recorded)'


def test_validation_reasons_measurements_and_report():
    state = RecognitionSelection(); state.replace(sample(False, with_extent=False))
    identity = state.snapshot.roots[0].children[0].children[1].id
    sections = inspector_sections(state.snapshot, identity)
    v = values(sections)
    assert v['Board validation'] == {'Status': 'Rejected', 'Reasons': 'termination unobservable'}
    assert v['Validation measurements'] == {'coverage': '0.0', 'valid': 'No'}
    assert 'Board validation\nStatus: Rejected\nReasons: termination unobservable' in property_report(sections)
    state.replace(sample(True))
    v = values(inspector_sections(state.snapshot, state.snapshot.roots[0].children[0].children[1].id))
    assert v['Board validation']['Status'] == 'Accepted'
    assert v['Board validation']['Reasons'] == '(none recorded)'


def test_invalid_and_obsolete_identity_and_unassociated_geometry():
    state = RecognitionSelection(); assert inspector_sections(None, None) == ()
    result = sample(); result.grids[0].detection_index = 5
    state.replace(result)
    old = state.snapshot.roots[1].id
    section = values(inspector_sections(state.snapshot, old))['Geometry result']
    assert section['Originating detection index (zero-based)'] == 'Unavailable'
    assert section['Stored detection index (zero-based)'] == '5'
    assert inspector_sections(state.snapshot, ResultId(object(), 'geometry', 0)) == ()
    state.replace(sample())
    assert inspector_sections(state.snapshot, old) == ()
    assert property_report(()) == ''
