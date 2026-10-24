/**
@file

Copyright <NAME> 2008

*/

#include <boost/python.hpp>

#include "bio/defs.h"
#include "bio/useradmin.h"
#include "bio/environment.h"
#include "biopsy/python.h"
#include "biopsy/analyse.h"


using namespace boost;
using namespace boost::python;
using namespace std;

namespace boost { namespace python { namespace indexing {

} } }


namespace biopsy {


struct environment
{
	static float BindingPrior() {
		return bio::BioEnvironment::singleton().get_tf_binding_prior();
	};

	static std::string data_dir() {
		return bio::BioEnvironment::singleton().data_dir;
	};

	static unsigned max_chain_max_num_sequences() {
		return get_max_chain_max_num_sequences();
	};

	static unsigned  transpath_major_version() {
		return  bio::BioEnvironment::singleton().transpath_major_version;
	}
	static unsigned  transpath_minor_version() {
		return  bio::BioEnvironment::singleton().transpath_minor_version;
	}
	static unsigned  transcompel_major_version() {
		return  bio::BioEnvironment::singleton().transcompel_major_version;
	}
	static unsigned  transcompel_minor_version() {
		return  bio::BioEnvironment::singleton().transcompel_minor_version;
	}
	static unsigned  transfac_major_version() {
		return  bio::BioEnvironment::singleton().transfac_major_version;
	}
	static unsigned  transfac_minor_version() {
		return  bio::BioEnvironment::singleton().transfac_minor_version;
	}

	static string custom_PSSM_version() {
		return  bio::BioEnvironment::singleton().custom_PSSM_version;
	}

};


void export_user()
{

	class_< bio::user_admin, noncopyable >( "UserAdmin", no_init )
		ADD_STATIC_PROPERTY(bio::user_admin,user)
		.add_static_property("userType",&bio::user_admin::userType)
		.add_static_property("isAllowed",&bio::user_admin::isAllowed)
		.def ("getCookie",&bio::user_admin::getCookie)
		.staticmethod( "getCookie" )
		.def ("validate",&bio::user_admin::validate)
		.staticmethod( "validate" )
		.def ("setPasswd",&bio::user_admin::setPasswd)
		.staticmethod( "setPasswd" );

	class_< environment, noncopyable >( "Environment", no_init )
		.add_static_property( "bindingPrior", &environment::BindingPrior )
		.add_static_property( "max_chain_max_num_sequences", &environment::max_chain_max_num_sequences )
		.add_static_property( "data_dir", &environment::data_dir)
		.add_static_property( "transfac_major_version", &environment::transfac_major_version )
		.add_static_property( "transfac_minor_version", &environment::transfac_minor_version )
		.add_static_property( "custom_PSSM_version", &environment::custom_PSSM_version );

}
}