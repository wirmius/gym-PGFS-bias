import os

# the root of this package
PGFS_MODULE_ROOT = os.path.abspath(os.path.dirname(__file__))

# folder with configuration files
CONFIG_ROOT = os.path.join(PGFS_MODULE_ROOT, "configs")

# dataset root for all the raw/downloaded data
DOWNLOAD_ROOT = os.path.abspath(os.path.join(os.path.expanduser('~'), '.chemPGFS'))

# a convention for the maximum verbosity of things
VERBOSITY_MAX = 3

# some strings for rendering svgs and strings
DEFAULT_MOL_WIDTH = 250
DEFAULT_MOL_HEIGHT = 250
DEFAULT_TEXT_HEIGHT = 80
DEFAULT_RXN_TEXT_HEIGHT = 80
DEFAULT_SCORE_TEXT_HEIGHT = 20
DEFAULT_ARROW_HEIGHT = 20
# additive templates - bireactant and monoreactant
additive_template_R2 = "{} + {}"
additive_template = "{} -{}-> {}"
# elements of the svg rendering
swag_base = """<?xml version='1.0' encoding='iso-8859-1'?>
<svg version='1.1' baseProfile='full'
              xmlns='http://www.w3.org/2000/svg'
                      xmlns:rdkit='http://www.rdkit.org/xml'
                      xmlns:xlink='http://www.w3.org/1999/xlink'
                  xml:space='preserve'
width='{width}px' height='{height}px' viewBox='0 0 {width} {height}'>
<!-- END OF HEADER -->
<rect style='opacity:1.0;fill:#FFFFFF;stroke:none' width='{width}' height='{height}' x='0' y='0'> </rect>
{content}
</svg>"""
swag_arrow = swag_base.format(width=250, height=DEFAULT_ARROW_HEIGHT, content = """
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="0" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#000" />
    </marker>
  </defs>
  <line x1="50" y1="10" x2="200" y2="10" stroke="#000" stroke-width="1" marker-end="url(#arrow)" />
""")
swag_rxn_text = swag_base.format(width=250,
                                 height=DEFAULT_RXN_TEXT_HEIGHT,
                                 content = """<text x="50" y="15" font-size="0.8em" fill="brown">{:30.30}</text>""")

swag_score = swag_base.format(width=250,
                              height=DEFAULT_SCORE_TEXT_HEIGHT,
                              content = """<text x="100" y="15" fill="blue">{:.4f}</text>""")

MOLDSET_SELECT_DESCRIPTORS = [
    "MaxEStateIndex",
    "MinEStateIndex", "MinAbsEStateIndex", "qed", "MolWt", "FpDensityMorgan1", "BalabanJ", "PEOE_VSA10", "PEOE_VSA11",
    "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "SMR_VSA7", "SlogP_VSA3",
    "SlogP_VSA5", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5", "EState_VSA6", "FractionCSP3",
    "MolLogP", "Kappa2", "PEOE_VSA2", "SMR_VSA5", "SMR_VSA6", "EState_VSA7", "Chi4v",
    "SMR_VSA10", "SlogP_VSA4", "SlogP_VSA6", "EState_VSA8", "EState_VSA9", "VSA_EState9"
]