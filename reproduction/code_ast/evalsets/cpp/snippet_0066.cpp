/*
 * MIT License
 * 
 * Copyright (c) 2017 aff3ct
 * Copyright (c) 2018 <NAME>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* 
 * File:   Codec.cpp
 * Author: simon
 * 
 * Created on September 13, 2019, 11:15 AM
 */

#include "Codec.hpp"

using namespace aff3ct;
using namespace aff3ct::simulation;

Codec::Codec() 
{
}

void Codec::printSocketInfo(std::ostream &stream, module::Socket &socket)
{
    stream << "  getName " << socket.get_name() << std::endl;
    stream << "  get_databytes " << socket.get_databytes() << std::endl;
    stream << "  get_datatype_size " << (int) socket.get_datatype_size() << std::endl;
    stream << "  get_datatype_string " << socket.get_datatype_string() << std::endl;
    stream << "  get_n_elmts " << socket.get_n_elmts() << std::endl;
}
void Codec::printSocketTypeInfo(std::ostream &stream, module::socket_t type)
{
    using namespace module;
    switch (type) {
        case socket_t::SIN :{
            stream << "  SIN" << std::endl;
            break;
        }
        case socket_t::SIN_SOUT :{
            stream << "  SIN_SOUT" << std::endl;
            break;
        }
        case socket_t::SOUT :{
            stream << "  SOUT" << std::endl;
            break;
        }
    }
}

int Codec::getK()
{
    
    if (creator) {
        factory::Codec_Generic::parameters & params = creator->getParams();
        
        return params.cdc->enc->K;
    }
    else
        return 0;
}

int Codec::getN() 
{
    if (creator) {
        factory::Codec_Generic::parameters & params = creator->getParams();
        return params.cdc->enc->N_cw;
    }
    else
        return 0;
}