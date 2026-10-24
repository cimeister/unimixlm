import { noMatchingLocation } from '../rejectionReasons';
import { filterByLocation } from './filterByLocation';
import { Message, Payload } from '../../../models/message';
import { LocatedMessage } from '../../../models/locatedMessage';
import mock from 'mock-fs';

describe('Filter by location', () => {
    beforeEach(() => {
        const locationFileContents = `[
          {
            "segment": {
              "code": "0000000",
              "fclass": "4",
              "lat": 40.03314777219885,
              "lon": -83.17328033080445,
              "name": "SOMEWHERE"
            },
            "intersection": {
              "name": "SOMEWHERE @ NOWHERE",
              "camera": "camera @ somewhere"
            }
          }]`;
        mock({
            'src/scripts/output/locations.json': locationFileContents,
        });
    });

    afterEach(() => {
        mock.restore();
    });

    it('rejects if no matching location', (done) => {
        const filterFn = filterByLocation();
        const payload = new Payload(0, 'noValidLocation', 'sometime', 0, 0);
        const message = new Message(payload);
        filterFn(message)
            .then(() => {
                fail('Promise not rejected');
            })
            .catch((error) => {
                expect(error).toContain(noMatchingLocation);
            })
            .then(done);
    });

    it('finds matching location', (done) => {
        const filterFn = filterByLocation();
        const payload = new Payload(0, '0000000', 'sometime', 0, 0);
        const message = new Message(payload);
        filterFn(message)
            .then((result: LocatedMessage) => {
                expect(result.segment.code).toEqual(payload.code);
                const expectedSegmentName = 'SOMEWHERE';
                expect(result.segment.name).toEqual(expectedSegmentName);
                const expectedIntersection = 'SOMEWHERE @ NOWHERE';
                expect(result.intersection.name).toEqual(expectedIntersection);
            })
            .then(done);
    });
});