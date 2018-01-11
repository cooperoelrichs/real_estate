from functools import reduce


def to_str_if_none(x):
    if x is None:
        return str(x)
    else:
        return x


class ObjectWithDictEquality(object):
    def __eq__(self, other):
        """Member wise equality."""
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        """Not `return not self.__eq__(other)`."""
        return not self == other


class Property(ObjectWithDictEquality):
    def __init__(self, sale_type, details, address_text):
        self.sale_type = sale_type
        self.details = details
        self.address_text = address_text
        self.state_and_postcode = NotYetPopulated()
        # self.address = NotYetPopulated()

        self.self_validation()

    def populate_address(self, address):
        raise RuntimeError('This is no longer supported.')

    def is_valid(self):
        return all(self.map_attributes(lambda x: x.is_valid()))

    def error_types(self):
        errors = self.map_attributes(lambda x: None if x.is_valid() else x)
        return [x for x in errors if x is not None]

    def error_type_name(self):
        # This relies on map_attributes use of ordered_attributes to assume
        # that the first error gets precedence
        return str(type(self.error_types()[0]))

    def to_tuple(self):
        tupled_tuples = self.map_attributes(lambda x: x.to_tuple())
        flatened_tuples = self.sum_tuples(tupled_tuples)
        return flatened_tuples

        return self.map_attributes(lambda x: x.to_tuple())

    def column_names(self):
        grouped_names = self.map_attributes(lambda x: type(x).column_names())
        flatened_names = self.sum_tuples(grouped_names)
        return flatened_names

    def summarise(self):
        return (
            'Property: \n - %s\n - %s\n - %s\n - %s' %
            self.map_attributes(lambda x: x.summarise())
        )

    def ordered_attributes(self):
        return [
            self.sale_type,
            self.details,
            self.address_text,
            self.state_and_postcode
        ]

    def map_attributes(self, fn):
        return tuple(map(fn, self.ordered_attributes()))

    def sum_tuples(self, tuples):
        return tuple(reduce(lambda y1, y2: y1 + y2, tuples))

    def self_validation(self):
        type_checks = [
            (self.sale_type, SaleType),
            (self.details, Details),
            (self.address_text, AddressText),
            (self.state_and_postcode, NotYetPopulated)
        ]
        self.type_check(type_checks)

    def type_check(self, checks):
        for attr, required_type in checks:
            if not isinstance(attr, required_type):
                raise TypeError('%s is not %s' %
                    (str(attr), str(required_type))
                )


class EmptyPropertyAttribute(ObjectWithDictEquality):
    def __init__(self):
        pass

    def is_valid(self):
        return False

    def summarise(self):
        return ''


class NotYetPopulated(EmptyPropertyAttribute):
    def not_yet_populated_error(self):
        raise RuntimeError('This field needs to be populated')

    def is_valid(self):
        self.not_yet_populated_error()

    def summarise(self):
        self.not_yet_populated_error()

    def column_names():
        self.not_yet_populated_error()

    def to_tuple(self):
        self.not_yet_populated_error()


class DataContentTypeNotSupported(Property):
    def __init__(self, data_content_type):
        self.data_content_type = data_content_type
        self.sale_type = EmptyPropertyAttribute()
        self.details = EmptyPropertyAttribute()
        self.address_text = EmptyPropertyAttribute()

    def is_valid(self):
        return False

    def summarise(self):
        return 'data-content-type not supported, %s' % self.data_content_type


class AddressText(ObjectWithDictEquality):
    def __init__(self, string):
        self.string = string

    def is_valid(self):
        return True

    def summarise(self):
        return self.string

    def column_names():
        return ('address_text',)

    def to_tuple(self):
        return (self.string,)


class Address(ObjectWithDictEquality):
    def __init__(self,
                 house, house_number, road, suburb, state, postcode,
                 address_is_valid):
        self.house = house
        self.house_number = house_number
        self.road = road
        self.suburb = suburb
        self.state = state
        self.postcode = postcode
        self.address_is_valid = address_is_valid

    def is_valid(self):
        return True

    def to_tuple(self):
        return (
            self.house,
            self.house_number,
            self.road,
            self.suburb,
            self.state,
            self.postcode,
            self.address_is_valid
        )

    def column_names(self):
        return (
            'house',
            'house_number',
            'road',
            'suburb',
            'state',
            'postcode',
            'address_is_valid'
        )

    def summarise(self):
        return (
            '%s, %s, %s, %s, %s %s, %s' %
            self.to_tuple()
        )


class AddressParseFailed(ObjectWithDictEquality):
    def __init__(self, string, components):
        self.string = string
        self.components = components

    def is_valid(self):
        return False

    def summarise(self):
        return ('Address parsing failed, %s, %s' %
            (self.string, str(self.components))
        )


class StateAndPostcode(ObjectWithDictEquality):
    def __init__(self, state, postcode):
        self.state = state
        self.postcode = postcode
        self.init_check()

    def is_valid(self):
        return True

    def to_tuple(self):
        return (
            self.state,
            self.postcode
        )

    def column_names():
        return (
            'state',
            'postcode'
        )

    def summarise(self):
        return (
            '%s, %s' % (self.state, str(self.postcode))
        )

    def init_check(self):
        if self.state is None:
            raise(RuntimeError('State is required.'))


class Details(ObjectWithDictEquality):
    def __init__(self, property_type, bedrooms, bathrooms, garage_spaces,
                 land_area, floor_area):
        self.property_type = property_type
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.garage_spaces = garage_spaces
        self.land_area = land_area
        self.floor_area = floor_area
        self.init_check()

    def is_valid(self):
        return True

    def to_tuple(self):
        return self.property_type.to_tuple() + (
            self.bedrooms,
            self.bathrooms,
            self.garage_spaces
        )

    def column_names():
        return PropertyType.column_names() + (
            'bedrooms',
            'bathrooms',
            'garage_spaces'
        )

    def summarise(self):
        return (
            '%s, %s bedrooms, %s bathrooms, %s garage spaces' %
            (self.property_type.name,
             str(self.bedrooms),
             str(self.bathrooms),
             str(self.garage_spaces))
        )

    def init_check(self):
        if self.property_type is None:
            raise(RuntimeError('Required Property Details are missing.'))


class SaleType(ObjectWithDictEquality):
    def __init__(self, name, prices, under_contract, under_application):
        self.name = name
        self.prices = prices
        self.under_contract = under_contract
        self.under_application = under_application

    def is_valid(self):
        return True

    def to_tuple(self):
        return (
            self.name,
            self.under_contract,
            self.under_application
        ) + self.prices_tuple()

    def prices_tuple(self):
        if self.prices is not None:
            return (min(self.prices), max(self.prices))
        else:
            return (None, None)

    def column_names():
        return (
            'sale_type',
            'under_contract',
            'under_application',
            'price_min',
            'price_max'
        )

    def summarise(self):
        return '%s $%s %s %s' % (
            self.name, self.join_prices(self.prices),
            str(self.under_contract),
            str(self.under_application)
        )

    def join_prices(self, prices):
        if prices is None:
            return None
        else:
            strings = [str(x) for x in prices]
            return ' and '.join(strings)


class PrivateTreaty(SaleType):
    def __init__(self, price, under_contract):
        super().__init__('Private Treaty', price, under_contract, None)


class OffPlan(SaleType):
    def __init__(self, price, under_contract):
        super().__init__('Off Plan', price, under_contract, None)


class Auction(SaleType):
    def __init__(self, under_contract):
        super().__init__('Auction', None, under_contract, None)


class Tender(SaleType):
    def __init__(self, under_contract):
        super().__init__('Tender', None, under_contract, None)


class Negotiation(SaleType):
    def __init__(self, under_contract):
        super().__init__('Negotiation', None, under_contract, None)


class ContactAgent(SaleType):
    def __init__(self, under_contract):
        super().__init__('Contact Agent', None, under_contract, None)


class SaleTypeParseFailed(SaleType):
    def __init__(self):
        super().__init__('Sale Type Parsing Failed', None, None, None)

    def is_valid(self):
        return False

    def summarise(self):
        return 'Sale Type parsing failed'


class UnableToFindSaleTypeText(SaleType):
    def __init__(self):
        super().__init__(
            'Parsing Failed, unable to find sale type text',
            None, None, None
        )

    def is_valid(self):
        return False

    def summarise(self):
        return 'Unable to find Sale Type text'


class Rental(SaleType):
    def __init__(self, price, under_application):
        super().__init__('Rental', price, None, under_application)


class RentalNegotiation(SaleType):
    def __init__(self, under_application):
        super().__init__('Rental by Negotiation', None, None,
                         under_application)


class RentalUnderApplication(SaleType):
    def __init__(self):
        super().__init__('Rental Under Application', None, None, True)


class RentalTypeParseFailed(SaleType):
    def __init__(self, sale_text):
        self.sale_text = sale_text
        super().__init__('Rental Type Parsing Failed', None, None, None)

    def is_valid(self):
        return False

    def summarise(self):
        return 'Rental Type parsing failed: %s' % self.sale_text


class PropertyType(ObjectWithDictEquality):
    def __init__(self, name):
        self.name = name

    def is_valid(self):
        return True

    def to_tuple(self):
        return (self.name,)

    def column_names():
        return ('property_type',)


class House(PropertyType):
    def __init__(self):
        super().__init__('House')


class TownHouse(PropertyType):
    def __init__(self):
        super().__init__('Town House')


class Unit(PropertyType):
    def __init__(self):
        super().__init__('Unit')


class ServicedApartment(PropertyType):
    def __init__(self):
        super().__init__('Serviced Apartment')


class UnitBlock(PropertyType):
    def __init__(self):
        super().__init__('UnitBlock')


class Studio(PropertyType):
    def __init__(self):
        super().__init__('Studio')


class Land(PropertyType):
    def __init__(self):
        super().__init__('Land')


class SemiRural(PropertyType):
    def __init__(self):
        super().__init__('Semi Rural')


class Duplex(PropertyType):
    def __init__(self):
        super().__init__('Duplex')


class RetirementLiving(PropertyType):
    def __init__(self):
        super().__init__('Retirement Living')


class Rural(PropertyType):
    def __init__(self):
        super().__init__('Rural')


class NotSpecified(PropertyType):
    def __init__(self, ):
        super().__init__('Not Specified')


class PropertyTypeNotSupported(PropertyType):
    def __init__(self, property_type_text, soup_with_href):
        self.property_type_text = property_type_text
        self.soup_with_href = soup_with_href
        super().__init__('Not Specified')

    def is_valid(self):
        return False

    def summarise(self):
        return ('Property type not supported, %s, %s' %
            (self.property_type_text, self.soup_with_href)
        )
