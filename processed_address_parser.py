from time import sleep
import multiprocessing
import real_estate.real_estate_property as rep


class PAP():
    def parse__(properties):
        address_strings = [p.address_text.string for p in properties]

        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=PAP.import_and_parse, args=(q, address_strings)
        )
        print('Starting separate process for parsing addresses.')
        p.start()
        r = PAP.get_results(p, q)
        p.join()
        print('Separate process finished.')

        PAP.ensure_queue_is_empty(q)
        properties = PAP.populate_addresses(properties, r)
        return properties

    def parse(properties):
        address_strings = [p.address_text.string for p in properties]
        PAP.dump_strings(address_strings)
        r = PAP.import_and_parse(None, address_strings)
        properties = PAP.populate_addresses(properties, r)
        return properties

    def parse_from_json():
        address_strings = PAP.load_strings()
        r = PAP.import_and_parse(None, address_strings)
        # properties = PAP.populate_addresses(properties, r)
        return r

    FN = 'testing.json'
    def dump_strings(strings):
        from real_estate.json_load_and_dump import JSONLoadAndDump
        print('dumping %i strings to %s.' % (len(strings), PAP.FN))
        JSONLoadAndDump.dump_to_file(strings, PAP.FN)

    def load_strings():
        from real_estate.json_load_and_dump import JSONLoadAndDump
        fn = 'testing.json'
        strings = JSONLoadAndDump.load_from_file(PAP.FN)
        print('loaded %i strings to %s.' % (len(strings), PAP.FN))
        return strings

    def get_results(p, q):
        r = None
        while True:
            sleep(0.1)
            q_empty = q.empty()
            p_is_alive = p.is_alive()
            if not q_empty:
                r = q.get()
                break
            elif not p_is_alive and r is not None:
                break
            elif not p_is_alive:
                raise RuntimeError(
                    'p died without adding to the queue - exit code was %i.'
                    % p.exitcode
                )
        return r

    def populate_addresses(properties, r):
        for p, components in zip(properties, r):
            p.populate_address(PAP.maybe_create_address(components))
        return properties

    def ensure_queue_is_empty(q):
        if q.empty() is False:
            raise RuntimeError('Some results were left in the Queue.')

    def import_and_parse(q, strings):
        print('Importing address parser.')
        from real_estate.address_parser import RealEstateAddressParser
        parser = RealEstateAddressParser()
        components = PAP.parse_addresses(parser, strings)
        print('Parsing complete.')
        # q.put(components)
        print('Added components to the queue.')
        return components

    def parse_addresses(parser, strings):
        return [parser.parse_and_validate_address(s) for s in strings]

    def maybe_create_address(address_components):
        if type(address_components) is rep.AddressParseFailed:
            return address_components
        else:
            return PAP.create_address(address_components)

    def create_address(address_components):
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
