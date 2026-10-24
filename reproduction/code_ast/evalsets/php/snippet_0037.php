<?php

namespace App\Http\Controllers\Admin\PropertyManage;

use Illuminate\Http\Request;
use App\Http\Controllers\Controller;
use App\Models\Countries\Country;
use App\Models\States\State;
use App\Models\Cities\City;
use App\Models\ZipCodes\ZipCode;
use App\Models\Properties\Property;
use Auth;
use Session;
use Illuminate\Support\Facades\Input;
use App\Models\Company_Managment\company_managment;
use DB;
class PropertyManageController extends Controller
{
public function __construct()
    {
        $this->middleware('auth');
    }


    public function index(Request $request)
    {
        dd($request);
        $property_id = $request['property_id'];
    	if($property_id){
		$property_id = $request['property_id'];
    	}
    	else{
    		$get_property_id = Property::first();
    		$property_id = $get_property_id['id'];
    	}
    	$where = array('id' => $property_id);
		$data['get_property_info'] = Property::with('country_list','state_list','city_list','zipcode_list')
		->where($where)->first();

    	$data['get_property_data'] = Property::all();
    	  $data_note = DB::table('propertiesnotes')
->select('propertynotesdate','propertynotestime','propertynotes','propertynotesattachment','id','propertyID')
->where(['propertyID' => $property_id])
->get();
        return view('admin.property_manage.index', $data,compact('data_note')); 
       }

       public function create(){
    	
    	$data['get_property_data'] = Property::all();
    	
    	$data['country']  = Country::pluck('countrysh', 'id');
        // $company_managment = company_managment::pluck("mgt_company","id");
         $company_managment = company_managment::where('adentcompany_id' ,'=', NULL)
    
       ->pluck("mgt_company","id");
    	return view('admin.property_manage.create', $data ,compact('company_managment'));
    }

       public function get_property_list(Request $request)
       {
		$get_property = $request->get_property;
		$where = array('id' => $get_property);
		$result = Property::with('country_list','state_list','city_list','zipcode_list')
		->where($where)->first();

	if($result){
			return Response()->json($result);
	    }

    }




    




    
    public function edit(Request $request){
    	$get_property = $request->id;
 
		$where = array('id' => $get_property);
    $get_property = Property::with('country_list','state_list','city_list','zipcode_list')
        ->where($where)->first();

        $idsArr = explode(',',$get_property['admininvoices']);
 
          
         $thiswhere =array('id' => $get_property);
        $get=Property::with('mangment_company')->where($where)->first();


    	$data['get_property_data'] = Property::all();
       $get_property_data1 = Property::where($where)->get();
    	$data['country']  = Country::pluck('countrysh', 'id');
        $company_managment = company_managment::pluck("mgt_company","id");
 
 
 
    	return view('admin.property_manage.property_edit', $data ,compact('company_managment','get_property','get','get_property_data1','idsArr'));
    }



    public function store(Request $request)
    {
         


        $user_id =  Auth::user()->id;


        $request->validate([

            'propertyName' => 'required',
            'propertyAd' => 'required',
            'propertyphone' => 'required|regex:/^([0-9\s\-\+\(\)]*)$/|min:10',
            'propertyFax' => 'required|regex:/^([0-9\s\-\+\(\)]*)$/|max:6',
            'propertyAltNo' => 'required',
            'email' => 'required|email',
            'mgt' => 'required',
            'country' => 'required',
            'state' => 'required',
            'city' => 'required',
            'zip' => 'required',
            'lastdatepic' =>'required',
            'abbrevation' =>'required',
            'invoices'   => 'required',
         
                ]);



$arrstring = implode(",", $request->invoices);
   $inputval['invoices'] = $arrstring;

              $data = new Property();
            $data->propertyName = $request->propertyName;
            $data->propertyAd = $request->propertyAd;
            $data->propertyphone = $request->propertyphone;
            $data->propertyFax = $request->propertyFax;
            $data->propertyAltNo = $request->propertyAltNo;
            $data->propertyEmail = $request->email;
            $data->MgmtCompID = $request->mgt;
            $data->country_id = $request->country;
            $data->state_id = $request->state;
            $data->city_id = $request->city;
            $data->zipcode_id = $request->zip;
            $data->lastcontract = $request->lastdatepic;
            $data->abbrevation = $request->abbrevation;
            $data->admininvoices = $inputval['invoices'];
            $data->user_id = $user_id;

 
    $data->save();
     Session::flash('message', 'Data has been Submit!'); 
    return redirect()->back();





    

    }

public function update(Request $request ,$id)
{

    // dd($id);
    $user_id =  Auth::user()->id;



        $request->validate([

            'propertyName' => 'required',
            'propertyAd' => 'required',
            'propertyphone' => 'required|regex:/^([0-9\s\-\+\(\)]*)$/|min:10',
            'propertyFax' => 'required|regex:/^([0-9\s\-\+\(\)]*)$/|max:6',
            'propertyAltNo' => 'required',
            'email' => 'required|email',
            'mgt' => 'required',
            'country' => 'required',
            'state' => 'required',
            'city' => 'required',
            'zip' => 'required',
            'lastdatepic' =>'required',
            'abbrevation' =>'required',
            'invoices'   => 'required',
    
                ]);





    $arrstring = implode(",", $request->invoices);
   $inputval['invoices'] = $arrstring;
    $data = Property::find($id);
      $data->propertyName = $request->propertyName;
            $data->propertyAd = $request->propertyAd;
            $data->propertyphone = $request->propertyphone;
            $data->propertyFax = $request->propertyFax;
            $data->propertyAltNo = $request->propertyAltNo;
            $data->propertyEmail = $request->email;
            $data->MgmtCompID = $request->mgt;
            $data->country_id = $request->country;
            $data->state_id = $request->state;
            $data->city_id = $request->city;
            $data->zipcode_id = $request->zip;
            $data->lastcontract = $request->lastdatepic;
            $data->abbrevation = $request->abbrevation;
            $data->admininvoices = $inputval['invoices'];
          
            $data->save();
             Session::flash('message', 'Data has been Updated!'); 
    return redirect()->back();
}
public function save_notes(Request $request){
    echo "string";
}
// public function notes(Request $request ,$id)
// {
//     echo "string"; die();
//      $product_id = $id;
// $user_id =  Auth::user()->id;
//     $arrstring = implode(",", $request->checkbox);
//    $inputval['checkbox'] = $arrstring;

     
//      $project = $request->file('file');
   
//       $new_name = rand() . '.' . $project->getClientOriginalExtension();
//       $project->move(public_path('notesfile'), $new_name);



//     $form_data = array(
//             'propertyID'   =>$product_id,
//             'propertynotesdate'       =>   $request->notedate,
//             'propertynotestime' =>   $request->note_time,
//             'propertynotes' =>   $request->expence_comment,
//             'propertybillnote' =>   $inputval['checkbox'],
//             'propertynotesattachment'    => $new_name,
//              'user_id'    =>   $user_id,
              
//         );
 


//      DB::table('propertiesnotes')->insert(array($form_data));
//      return response()->json(['message' => 'Uploaded successfully']);
// }

  public function deletenotes(Request $request){
        $delete_charge_type = $request->delete_charge_type;


$result = DB::table('propertiesnotes')->where('id', $delete_charge_type)->delete();
        
     
        return response()->json([
                    'message'   => 'Notes Delete Successfully',
                    'data' => $result,
                    'class_name'  => 'alert alert-success'
                    ]);

    }

}