/*
 *  Copyright 2019 Carnegie Technologies
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

extern "C"
{
#include <net/if.h>
}

#include "../shared/SocketApi.cpp"

using namespace Pravala;

ERRCODE SocketApi::bindToIface ( int sockFd, SocketType sockType, const String & ifaceName )
{
    if ( sockType != SocketApi::SocketDgram4
         && sockType != SocketApi::SocketDgram6
         && sockType != SocketApi::SocketStream4
         && sockType != SocketApi::SocketStream6 )
    {
        return Error::InvalidParameter;
    }

    if ( ifaceName.isEmpty() )
    {
        return Error::TooShortName;
    }

    const int ifIndex = if_nametoindex ( ifaceName.c_str() );

    if ( ifIndex < 1 )
    {
        return Error::NotFound;
    }

    const bool isV4 = ( sockType == SocketApi::SocketStream4 || sockType == SocketApi::SocketDgram4 );

    if ( !SocketApi::setOption ( sockFd,
                                 isV4 ? IPPROTO_IP : IPPROTO_IPV6,
                                 isV4 ? IP_BOUND_IF : IPV6_BOUND_IF,
                                 ifIndex ) )
    {
        return Error::BindFailed;
    }

    return Error::Success;
}