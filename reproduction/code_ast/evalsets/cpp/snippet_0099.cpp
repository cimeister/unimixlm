#include "config/LogglyWriterConfig.h"
#include "JetStream.h"

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

namespace jetstream{
namespace config {

    LogglyWriterConfig::LogglyWriterConfig(JetStream *jetstream)
            : JetStreamConfig(jetstream) {


    }


    void LogglyWriterConfig::printHelp() {

		cerr << "Usage: jetstream loggly [OPTION]...\n"
				"Write from a kafka topic to loggly (batch api).\n"
				"\n"
				"Mandatory arguments to long options are mandatory for short options too.\n"
				"  -b, --brokers [BROKERS]                a csv list of kafka brokers\n"
				"                                         (optional; defaults to ENV JETSTREAM_BROKERS)\n"
				"  -c, --consumer-group [CONSUMER_GROUP]  the kafka consumer group shared among all consumers\n"
				"                                         (optional; defaults to ENV JETSTREAM_CONSUMER_GROUP)\n"
				"  -t, --topic [TOPIC]                    a destination kafka topic\n"
				"                                         (optional; defaults to ENV JETSTREAM_TOPIC)\n"
				"  -p, --product-code [PRODUCT_CODE]      a code identifying a part of your organization or product\n"
				"                                         (optional; defaults to ENV JETSTREAM_PRODUCT_CODE)\n"
				"  -h, --hostname [HOSTNAME]              the name of this host that will appear in the log entries\n"
				"                                         (optional; defaults to ENV JETSTREAM_HOSTNAME)\n"
				"  --token [LOGGLY_TOKEN]                the API token provided by loggly for your account\n"
				"                                         (required; defaults to ENV JETSTREAM_DESTINATION_TOKEN)\n"
				"  -ph, --prom-hostname [HOSTNAME]        prometheus push gateway hostname (eg. 'push.prom.com:5000')\n"
				"                                         (optional; defaults to ENV JETSTREAM_PROM_HOSTNAME)\n"
				"  -ps, --prom-secure [SECURE]            whether to use TLS for prometheus push gateway\n"
				"                                         eg. ('true' or 'false'). Defaults to 'false'. \n"
				"                                         (optional; defaults to ENV JETSTREAM_PROM_SECURE)\n"
		<< endl;

    }

}
}