/*
 * File: IA20-CommonLib/src/commonlib/net/mcast/Base.cpp
 *
 * Copyright (C) 2020, <NAME>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Base.h"

#include <ia20/commonlib/commonlib.h>
#include <unistd.h>

namespace IA20 {
namespace Net {
namespace Conn {
namespace UDP {

/*************************************************************************/
Base::Base(unsigned int iPort):
	iPort(iPort),
	iTimeout(C_UnLimited){
	IA20_TRACER;

	if((iSocket=socket(AF_INET, SOCK_DGRAM, 0)) < 0)
		IA20_THROW(SystemException("UDP mulitcast socket, port: ")<<(int)iPort);

	IA20_LOG(LogLevel::INSTANCE.isInfo(),"iSocket:"<<iSocket);
}
/*************************************************************************/
Base::~Base() throw(){
	IA20_TRACER;

	if(iSocket > 0)
		::close(iSocket);
}
/*************************************************************************/
void Base::setTimeout(int iTimeout){
	IA20_TRACER;

	if(iTimeout != C_UnLimited && iTimeout < 0)
		IA20_THROW(BadUsageException("timeout: ")<<iTimeout);

	this->iTimeout=iTimeout;

	IA20_LOG(LogLevel::INSTANCE.isInfo(),"fd("<<iSocket<<"), timeout: "<<this->iTimeout);
}
/*************************************************************************/
bool Base::waitForData(WaitMode iMode){
	IA20_TRACER;

	 fd_set set;
	 struct timeval timeout;

	 FD_ZERO(&set);
	 FD_SET(iSocket, &set);

	 timeout.tv_sec = 0;
	 timeout.tv_usec = 1000*iTimeout;

	 IA20_LOG(LogLevel::INSTANCE.isInfo(),"fd("<<iSocket<<"), timeout: "<<iTimeout);

	 int iRC = ::select(iSocket + 1,
			 	 (iMode == WM_Read  ? &set : NULL),
				 (iMode == WM_Write ? &set : NULL),
				 &set,
				 iTimeout == C_UnLimited ? NULL : &timeout);

	 IA20_LOG(LogLevel::INSTANCE.isInfo(),"rc="<<iRC);

	 switch(iRC){
	 	 case -1:
	 		 IA20_THROW(SystemException("select: ")<<iSocket)
			 break;
	 	 case 0:
	 		 return false;
	 		 break;
	 }

	 return true;
}
/*************************************************************************/
}
}
}
}