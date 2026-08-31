//
//  Copyright (C) 2012 <NAME>
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//

#include <iostream>
#include <fstream>
#include <exception>
#include <iomanip> // for setiosflags, setw

#include "midilib/include/midi_parser.hpp"
#include "midilib/include/midi_multiplexer.hpp"
#include "midilib/include/timed_midi_visitor.hpp"
///
/// This class only reacts on meta events. If the meta event is of type "text" (0x01)
/// then it will print the data of the event to the given output stream.
/// backward- and forward slashes will be converted to newlines.
/// texts that start with @ will be ignored.
///
struct print_text_visitor: public events::timed_visitor<print_text_visitor>
{
    typedef events::timed_visitor< print_text_visitor> parent;
    using parent::operator();

    print_text_visitor( std::ostream &output, midi_header &header)
        : output(output), parent( header)
    {
    }

    void operator()( const events::meta &event)
    {
        using namespace std;
        // is it a text event?
        // we're ignoring lyrics (0x05) events, because text events have more
        // information (like 'start of new line')
        if (event.type == 0x01)
        {
            string event_text( event.bytes.begin(), event.bytes.end());
            if (event_text.size() > 0 && event_text[0] != '@')
            {
                if (event_text[0] == '/' || event_text[0] == '\\')
                {
                    output << "\n" << setiosflags( ios::right) << setprecision(2) << fixed << setw( 6) << get_current_time() << '\t';
                    output << event_text.substr( 1);
                }
                else
                {
                    output << event_text;
                }
            }
        }
    }


private:
    std::ostream &output;
};

int main( int argc, char *argv[])
{
    using namespace std;
    if (argc != 2)
    {
        cerr << "usage: miditool <midi file name>\n";
        exit( -1);
    }

    try
    {
        string filename( argv[1]);
        ifstream inputfile( filename.c_str(), ios::binary);
        if (!inputfile)
        {
            throw runtime_error( "could not open " + filename + " for reading");
        }


        midi_file midi;
        if (!parse_midifile( inputfile, midi))
        {
            throw runtime_error( "I can't parse " + filename + " as a valid midi file");
        }

        // combine all midi tracks into one virtual track.
        midi_multiplexer multiplexer( midi.tracks);

        // now print all lyrics in the midi file by sending a print_text_visitor into the data structure.
        multiplexer.accept( print_text_visitor( cout, midi.header));
    }
    catch (const exception &e)
    {
        cerr << "something went wrong: " << e.what() << '\n';
    }

    return 0;
}