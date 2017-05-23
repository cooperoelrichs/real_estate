from postal import parser
import re
import real_estate.real_estate_property as rep


class RealEstateAddressParser(object):
    """
    Parse real estate address strings from the internet using libpostal.

    Wraps some pre and post processing around libpostal to improve results
    for Australian freeform real estate addresses from the internet.
    libpostal: https://github.com/openvenues/libpostal
    pypostal: https://github.com/openvenues/pypostal
    """

    PREPROCESSING_REGEX_SUBSTITUTIONS = [(re.compile(a), b) for a, b in [
        (r'  ', r' '),

        # libpostal doesn't like '&' between street names.
        (r'([a-zA-Z]+)(?: \& )([a-zA-Z]+)', r'\1 and \2'),

        # Ending with a partial postcode and '...'.
        (r'(.*)(?:\d{2}\.{3})$', r'\1'),

        # Ending with '...'.
        (r'(.*)(?:\.{3})$', r'\1'),

        # Special case. Remove '. '.
        (r'^(\. )', r''),

        # Remove "'" and '/' from property names.
        (r'\'([a-zA-Z ]+)\'/', r'\1 '),
        (r'\'([a-zA-Z ]+)\'', r'\1'),

        # Special case. Fix a typo
        (r'^(\d+) (-\d+)', r'\1\2'),

        # Special case. Fix typos
        (r'([a-zA-Z ]+)/blocks ([0-9]+)- section ([0-9]+), (?i)',
         r'\1 block __\2__ section __\3__, __no_street__ street, '),

        # Address doesn't have a street which confuses libpostal, so we
        # insert a fake street that will be removed after parsing.
        (r'block ([a-zA-Z0-9]+) section ([a-zA-Z0-9]+)( [a-zA-Z]+)?, (?i)',
         r'block __\1__ section __\2__\3, __no_street__ street, '),

        # Special case. Insert a fake street to help libpostal.
        (r'^(?:\* |\*/\* )\(no street name\), (?i)',
         r'__no_street__ street, '),

        # Special case.
        (r'^(.*), address available on request$(?i)',
         r'__address_available_on_request__ street, \1'),

        # libpostal doesn't understand the 'unit' prefix.
        (r'^units? (\d+), (?i)', r'\1/'),

        # Special case, Taggerty Steet without a street name.
        (r'(\d+ taggerty), (ngunnawal)(?i)', r'\1 Street, \2'),
        (r'(\d+ yerradhang), (ngunnawal)(?i)', r'\1 Street, \2'),
        (r'(\d+ bunda), (city)(?i)', r'\1 Street, \2'),
        (r'(\d+ chaseling), (phillip)(?i)', r'\1 Street, \2'),
        (r'(\d+ constitution), (reid)(?i)', r'\1 Avenue, \2'),

        # Special case, Totterdell is a loop and a street in Belconnen.
        (r'(1-9 totterdell), (belconnen)(?i)', r'\1 Street, \2'),

        # Certain city names confuse libpostal, turning ACT into a road.
        (r'(franklin), act (\d+)(?i)', r'\1, __ACT__ \2'),
        (r'(greenway), act (\d+)(?i)', r'\1, __ACT__ \2'),
        (r'(narrabundah), act (\d+)(?i)', r'\1, __ACT__ \2'),
        (r'(chisholm), act (\d+)(?i)', r'\1, __ACT__ \2'),

        # As above, but more dificult to fix than the others.
        (r'(melba)(, act \d+)(?i)', r'__\1_City__\2'),
        (r'(bonner)(, act \d+)(?i)', r'__\1_City__\2'),
        (r'(tharwa)(, act \d+)(?i)', r'__\1_City__\2'),
        (r'(uriarra) (village)(, act \d+)(?i)', r'__\1_\2_City__\3'),

        # Special cases.
        (
            r'^(\d+/\d+) manhattan on the park ([a-zA-Z]+)(?i)',
            r'__manhattan_on_the_park__ \1 \2'
        ),
        (r'^(nibu) - (\d+)(?i)', r'__\1__ \2'),
        (r'^(\d+ [a-zA-Z ]+) (form), (?i)', r'__\2__ \1, '),
        (r'^(\d+ [a-zA-Z ]+) (hudson) (square), (?i)', r'__\2_\3__ \1, '),
        (r'^(\d+) (mosaic),(\d+)(?i)', r'__\2__ \1/\3'),
    ]]

    POSTPROCESSING_LAMBDAS = [
        # Remove fake streets. Perserve order, remove all occurances (not a
        # set operation).
        (lambda x: [a for a in x if a != ('__no_street__ street', 'road')]),

        (lambda x: [
            (re.sub('__', '', a), b)
            if re.match(
                r'.*block __[a-z0-9]+__ section __[a-z0-9]+__', a
            ) is not None
            else (a, b)
            for a, b in x
        ])
    ]

    POSTPROCESSING_SUBSTITUTIONS = [
        # Turn ACT back into a state for certain special cases.
        (('__act__', 'road'), ('act', 'state')),
        (('__act__', 'city'), ('act', 'state')),

        # Special cases.
        (('__nibu__', 'road'), ('nibu', 'house')),
        (('__form__', 'road'), ('form', 'house')),
        (('__hudson_square__', 'house'), ('hudson square', 'house')),
        (('__mosaic__', 'road'), ('mosaic', 'house')),
        (('__melba_city__', 'city'), ('melba', 'city')),
        (('__bonner_city__', 'city'), ('bonner', 'city')),
        (('__tharwa_city__', 'city'), ('tharwa', 'city')),
        (('__uriarra_village_city__', 'city'), ('uriarra village', 'city')),
        (('monash', 'state_district'), ('monash', 'city')),
        (('tuggeranong', 'state_district'), ('tuggeranong', 'city')),
        (
            ('__manhattan_on_the_park__', 'house'),
            ('manhattan on the park', 'house')
        ),
        (
            ('__address_available_on_request__ street', 'road'),
            ('address available on request', 'special')
        ),
    ]

    STREET_NAMES_REQUIRING_FIXES = [
        # Certain road names confuse libpostal.
        # TODO Figure out how to make this general.
        'macfarlane burnet avenue',
        'newman morris circuit',
        'rolph place',
        'hesel tine street',
        'millhouse crescent',
        'lads place',
        'hobday place',
        'buckmaster crescent',
        'tarrant crescent',
        'greg urwin circuit',
        'fleay place',
        'barraclough crescent',
        'quinlivan crescent',
        'temple terrace',
        'jeff snell crescent',
        'menzel crescent',
        'glenbawn place',
        'nullarbor avenue',
        'elia ware crescent',
        'elphick place',
        'lawrenson circuit',
        'shirley taylor place',
        'pedrail place',
        'ulysses circuit',
        'flower place',
        'chinner crescent',
        'medley st',
        'bambridge crescent',
        'arden place',
        'florence taylor street',
        'ern florence crescent',
        'New South Wales Crescent',
        'boboyan road',
        'elizabeth jolley crescent'
    ]

    def parse_and_validate_address(self, address_string):
        address_components = self.parse_address(address_string)
        valid = AddressComponentValidator().validate_address_components(
            address_string, address_components
        )
        if valid:
            return address_components
        else:
            return rep.AddressParseFailed(address_string, address_components)

    def parse_address(self, address_string):
        address_string = self.preprocess_string(address_string)
        address_components = parser.parse_address(
            address_string,
            language='en', country='au'
        )
        address_components = self.postprocess_components(address_components)
        return address_components

    def preprocess_string(self, address_string):
        for regex, replacment in self.PREPROCESSING_REGEX_SUBSTITUTIONS:
            address_string = regex.sub(replacment, address_string)
        address_string = self.apply_street_name_preprocessing_fixes(
            address_string)
        return address_string

    def apply_street_name_preprocessing_fixes(self, address_string):
        regex_fixes = [
            (r'%s, (?i)' % x, self.street_fix_format(x) + ', ')
            for x in self.STREET_NAMES_REQUIRING_FIXES
        ]
        for regex, replacment in regex_fixes:
            address_string = re.sub(regex, replacment, address_string)
        return address_string

    def street_fix_format(self, string):
        return r'__%s__ street' % re.compile(' ').sub('_', string)

    def postprocess_components(self, address_components):
        for fn in self.POSTPROCESSING_LAMBDAS:
            address_components = fn(address_components)

        for match, substitute in self.POSTPROCESSING_SUBSTITUTIONS:
            address_components = [
                substitute if a == match else a for a in address_components
            ]
        address_components = self.apply_street_name_postprocessing_fixes(
            address_components)
        return address_components

    def apply_street_name_postprocessing_fixes(self, address_components):
        substitutions = [
            ((self.street_fix_format(x), 'road'), (x, 'road'))
            for x in self.STREET_NAMES_REQUIRING_FIXES
        ]
        for match, substitute in substitutions:
            address_components = [
                substitute if a == match else a for a in address_components
            ]
        return address_components


class AddressComponentValidator():
    REQUIRED_ADDRESS_COMPONENTS = [
        ['state', 'special'],
        ['postcode', 'special'],
        ['city', 'suburb']
    ]

    def validate_address_components(self, string, components):
        checks = [
            self.check_for_duplicates(string, components),
            self.check_for_suburb_and_city(string, components),
            self.check_for_required_components(string, components)
        ]

        return all(checks)

    def component_names(self, components):
        return [x for _, x in components]

    def check_for_duplicates(self, string, components):
        component_names = self.component_names(components)
        return len(component_names) == len(set(component_names))

    def check_for_suburb_and_city(self, string, components):
        component_names = self.component_names(components)
        return not (
            'suburb' in component_names and
            'city' in component_names
        )

    def check_for_required_components(self, string, components):
        component_names = self.component_names(components)
        results = []
        for requires_one_of_these in self.REQUIRED_ADDRESS_COMPONENTS:
            checks = [
                component_type in component_names
                for component_type in requires_one_of_these
            ]
            results.append(any(checks))

        return all(results)
