
#include <cpp11.hpp>
using namespace cpp11;

#include <vector>

#include "GeomagnetismHeader.h"

class WMMMagneticModel {
public:
    WMMMagneticModel(MAGtype_MagneticModel* model): model_(model) {}
    WMMMagneticModel(int num_terms): model_(nullptr) {
        model_ = MAG_AllocateModelMemory(num_terms);
        if (model_ == nullptr) {
            stop("Failed to MAGtype_MagneticModel with %d terms", num_terms);
        }
    }

    MAGtype_MagneticModel* model() {
        return model_;
    }

    ~WMMMagneticModel() {
        if (model_ != nullptr) {
            MAG_FreeMagneticModelMemory(model_);
        }
    }

private:
    MAGtype_MagneticModel* model_;
};



[[cpp11::register]]
std::string cpp_mm_version() {
    return VERSIONDATE_LARGE;
}

[[cpp11::register]]
SEXP cpp_mm_read_coef(std::string filename_utf8) {
    // some peculiarities with the input...filename isn't modified and
    // the models array has a hard-to-replicate type
    MAGtype_MagneticModel* models_sane[1];
    models_sane[0] = nullptr;

    char* filename_buf = (char*) filename_utf8.c_str();

    int result = MAG_robustReadMagModels(filename_buf, (MAGtype_MagneticModel**) models_sane, 1);
    if (result == 0) {
        if (models_sane[0] != nullptr) {
            MAG_FreeMagneticModelMemory(models_sane[0]);
        }
        stop("Failed to open model coefficients '%s'", filename_utf8.c_str());
    }

    return external_pointer<WMMMagneticModel>(new WMMMagneticModel(models_sane[0]));
}

[[cpp11::register]]
SEXP cpp_mm_read_coef_sv(std::string filename_utf8, std::string filename_sv_utf8) {
    MAGtype_MagneticModel *models_sane[1];
    models_sane[0] = nullptr;

    char* filename_buf = (char*) filename_utf8.c_str();
    char* filename_sv_buf = (char*) filename_sv_utf8.c_str();

    int result = MAG_robustReadMagneticModel_Large(filename_buf, filename_sv_buf, models_sane);
    if (result == 0) {
        if (models_sane[0] != nullptr) {
            MAG_FreeMagneticModelMemory(models_sane[0]);
        }
        stop(
            "Failed to open model coefficients from '%s' or '%s'",
            filename_utf8.c_str(),
            filename_sv_utf8.c_str()
        );
    }

    return external_pointer<WMMMagneticModel>(new WMMMagneticModel(models_sane[0]));
}

[[cpp11::register]]
void cpp_mm_coalesce_for_emm2017(SEXP mutable_model_sexp, SEXP model_sexp, bool zero_sv) {
    external_pointer<WMMMagneticModel> mutable_model = mutable_model_sexp;
    external_pointer<WMMMagneticModel> model = model_sexp;

    // make sure the model knows which epoch it is in
    mutable_model->model()->epoch = model->model()->epoch;

    // assign coefficients from the epoch model against the "main" model (2017)
    MAG_AssignMagneticModelCoeffs(
        mutable_model->model(), model->model(),
        model->model()->nMax, model->model()->nMaxSecVar
    );

    // For each set of coefficients that isn't the main model, some secular
    // variation coefficients need to be zeroed out
    if (zero_sv) {
        for(int i = 0; i < 16; i++) {
            int index = 16 * 17 / 2 + i;
            mutable_model->model()->Secular_Var_Coeff_G[index] = 0;
            mutable_model->model()->Secular_Var_Coeff_H[index] = 0;
        }
    }
}

[[cpp11::register]]
doubles cpp_mm_ellipsoidal_height(list coords, integers geoid_ints) {
    doubles lambda = coords["lambda"];
    doubles phi = coords["phi"];
    doubles height = coords["height"];

    if (geoid_ints.size() != 1038961) {
        stop("Expected `geoid` as an integer vector with length 1038961");
    }

    // geoid_ints are the float values scaled by 1000, but we need a float
    // buffer
    std::vector<float> geoid_floats(geoid_ints.size());
    for (R_xlen_t i = 0; i < geoid_ints.size(); i++) {
        geoid_floats[i] = geoid_ints[i] / 1000.0;
    }

    MAGtype_Ellipsoid ellipsoid;
    MAGtype_Geoid geoid;
    MAG_SetDefaults(&ellipsoid, &geoid);
    // double make sure the geoid *is* getting used here
    geoid.UseGeoid = 1;

    geoid.GeoidHeightBuffer = geoid_floats.data();
    geoid.Geoid_Initialized = 1;

    R_xlen_t size = lambda.size();
    writable::doubles height_ellipsoid(size);
    MAGtype_CoordGeodetic coord_geod;
    coord_geod.UseGeoid = 1;

    for (R_xlen_t i = 0; i < size; i++) {
        coord_geod.lambda = lambda[i];
        coord_geod.phi = phi[i];
        coord_geod.HeightAboveGeoid = height[i];

        // MAG_ConvertGeoidToEllipsoidHeight() crashes with NAs
        if (is_na(coord_geod.lambda) ||
              is_na(coord_geod.phi) ||
              is_na(coord_geod.HeightAboveGeoid)) {
            height_ellipsoid[i] = NA_REAL;
            continue;
        }

        MAG_ConvertGeoidToEllipsoidHeight(&coord_geod, &geoid);
        height_ellipsoid[i] = coord_geod.HeightAboveEllipsoid;
    }

    return height_ellipsoid;
}

[[cpp11::register]]
list cpp_mm_extract(SEXP model_sexp, list coords) {
    external_pointer<WMMMagneticModel> model = model_sexp;

    doubles lambda = coords["lambda"];
    doubles phi = coords["phi"];
    doubles height = coords["height"];
    doubles year = coords["year"];

    if (model->model() == nullptr) {
        stop("`model` is nullptr");
    }

    int nMax = model->model()->nMax;
    WMMMagneticModel timed_model((nMax + 1) * (nMax + 2) / 2);

    R_xlen_t size = lambda.size();
    writable::doubles decl(size);
    writable::doubles incl(size);
    writable::doubles decl_err(size);
    writable::doubles incl_err(size);

    MAGtype_Ellipsoid ellipsoid;
    MAGtype_CoordSpherical coord_spherical;
    MAGtype_CoordGeodetic coord_geod;
    coord_geod.HeightAboveGeoid = NA_REAL;
    coord_geod.UseGeoid = 0;
    MAGtype_Date user_date;
    MAGtype_GeoMagneticElements elements, element_errors;
    MAGtype_Geoid geoid;

    MAG_SetDefaults(&ellipsoid, &geoid);
    // double make sure the geoid isn't getting used here
    geoid.UseGeoid = 0;

    for (R_xlen_t i = 0 ; i < size; i++) {
        // for EMM this calculation is very slow, so check frequently
        if (((i + 1) % 10) == 0) {
            check_user_interrupt();
        }

        coord_geod.lambda = lambda[i];
        coord_geod.phi = phi[i];
        coord_geod.HeightAboveEllipsoid = height[i];
        user_date.DecimalYear = year[i];

        MAG_GeodeticToSpherical(ellipsoid, coord_geod, &coord_spherical);
        MAG_TimelyModifyMagneticModel(user_date, model->model(), timed_model.model());
        MAG_Geomag(ellipsoid, coord_spherical, coord_geod, timed_model.model(), &elements);
        MAG_CalculateGridVariation(coord_geod, &elements);
        MAG_WMMErrorCalc(elements.H, &element_errors);

        decl[i] = elements.Decl;
        incl[i] = elements.Incl;
        decl_err[i] = element_errors.Decl;
        decl_err[i] = element_errors.Incl;
    }

    writable::list out = {decl, incl, decl_err, incl_err};
    out.names() = {"decl", "incl", "decl_err", "incl_err"};
    return out;
}

// for IGRF13, the coefficients themselves are interpolated between
// five-year periods
void mm_igrf13_interpolated(MAGtype_MagneticModel* mutable_model,
                                const MAGtype_MagneticModel* model1,
                                const MAGtype_MagneticModel* model2,
                                double year) {
    // a number between zero and one used for interpolating
    double interp = (year - model1->epoch) / (model2->epoch - model1->epoch);

    int index;
    double g1, g2, h1, h2;
    for(int n = 1; n <= model1->nMax; n++) {
        for(int m = 0; m <= n; m++) {
            index = (n * (n + 1) / 2 + m);
            g1 = model1->Main_Field_Coeff_G[index];
            g2 = model2->Main_Field_Coeff_G[index];
            h1 = model1->Main_Field_Coeff_H[index];
            h2 = model2->Main_Field_Coeff_H[index];
            mutable_model->Main_Field_Coeff_G[index] = g1 + (g2 - g1) * interp;
            mutable_model->Main_Field_Coeff_H[index] = h1 + (h2 - h1) * interp;
            mutable_model->Secular_Var_Coeff_G[index] = g1 + (g2 - g1) * interp;
            mutable_model->Secular_Var_Coeff_H[index] = h1 + (h2 - h1) * interp;
        }
    }

    mutable_model->epoch = year;
}

[[cpp11::register]]
list cpp_mm_igrf13_extract(SEXP mutable_model_sexp, SEXP model1_sexp, SEXP model2_sexp,
                           list coords) {
    external_pointer<WMMMagneticModel> model = mutable_model_sexp;
    external_pointer<WMMMagneticModel> model1 = model1_sexp;
    external_pointer<WMMMagneticModel> model2 = model2_sexp;

    doubles lambda = coords["lambda"];
    doubles phi = coords["phi"];
    doubles height = coords["height"];
    doubles year = coords["year"];

    if (model->model() == nullptr || model1->model() == nullptr || model2->model() == nullptr) {
        stop("One of `model`, `model1`, or `model2` is nullptr");
    }

    R_xlen_t size = lambda.size();
    writable::doubles decl(size);
    writable::doubles incl(size);
    writable::doubles decl_err(size);
    writable::doubles incl_err(size);

    MAGtype_Ellipsoid ellipsoid;
    MAGtype_CoordSpherical coord_spherical;
    MAGtype_CoordGeodetic coord_geod;
    coord_geod.HeightAboveGeoid = NA_REAL;
    coord_geod.UseGeoid = 0;
    MAGtype_Date user_date;
    MAGtype_GeoMagneticElements elements;
    MAGtype_Geoid geoid;

    MAG_SetDefaults(&ellipsoid, &geoid);
    // double make sure the geoid isn't getting used here
    geoid.UseGeoid = 0;

    for (R_xlen_t i = 0 ; i < size; i++) {
        if (((i + 1) % 1000) == 0) {
            check_user_interrupt();
        }

        coord_geod.lambda = lambda[i];
        coord_geod.phi = phi[i];
        coord_geod.HeightAboveEllipsoid = height[i];
        user_date.DecimalYear = year[i];

        MAG_GeodeticToSpherical(ellipsoid, coord_geod, &coord_spherical);
        mm_igrf13_interpolated(
            model->model(),
            model1->model(), model2->model(),
            user_date.DecimalYear
        );
        MAG_Geomag(ellipsoid, coord_spherical, coord_geod, model->model(), &elements);
        MAG_CalculateGridVariation(coord_geod, &elements);

        decl[i] = elements.Decl;
        incl[i] = elements.Incl;
    }

    writable::list out = {decl, incl};
    out.names() = {"decl", "incl"};
    return out;
}