import real_estate.real_estate_property as rep
from real_estate.address_parser import RealEstateAddressParser


class AddressFactory(object):
    def __init__(self):
        self.parser = RealEstateAddressParser()

    def parse_address(self, address_text):
        address_components = self.parser.parse_and_validate_address(address_text)
        if type(address_components) is rep.AddressParseFailed:
            return address_components
        else:
            return self.create_address(address_components)

    def create_address(self, address_components):
        named_components = dict([(b, a) for a, b in address_components])
        address = rep.Address(
            house=named_components.get('house'),
            house_number=named_components.get('house_number'),
            road=named_components.get('road'),
            suburb=named_components.get('suburb',
                                        named_components.get('city')),
            state=named_components.get('state'),
            postcode=named_components.get('postcode')
        )
        return address
