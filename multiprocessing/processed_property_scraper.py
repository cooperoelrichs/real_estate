from time import sleep
import multiprocessing
import queue

from real_estate.website_scraper import WebsiteScraper


# Todo: joinable queue?

class PPS():
    def make_properties(q1, q2, assistant):
        pages = assistant.load_pages()
        properties = assistant.scraper.scrape_pages(pages)
        num_properties = len(properties)
        q1.put(num_properties)
        for i, v in enumerate(properties):
            sleep(0.01)
            # q2.put((i, v), timeout=60)
            try:
                q2.put_nowait((i, v))
            except:
                print('Put failed at number %i.' % i)
                raise

    def get_results(p, q):
        results = []
        while True:
            # sleep(0.01)
            p_is_alive = p.is_alive()
            try:
                i, r = q.get_nowait()
                results.append(r)
                # print(i)
            except queue.Empty:
                if not p_is_alive:
                    print('BREAKING at number %i' % i)
                    break
                sleep(0.1)
        return results

    def processed_make_properties(assistant):
        q1 = multiprocessing.Queue()
        q2 = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=Kip.make_properties, args=(q1, q2, assistant)
        )
        p.start()
        num_properties = q1.get()
        r = Kip.get_results(p, q2)
        p.join()

        if len(r) != num_properties:
            raise RuntimeError(
                "Didn't recieve all of the properties - %i of %i"
                % (len(r), num_properties)
            )
        print((len(r), num_properties))
        return r
