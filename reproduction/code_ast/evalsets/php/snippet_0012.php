<?php

namespace App\Http\Controllers;

use App\Setting;
use Symfony\Component\HttpFoundation\Session\Session;

class SettingsController extends Controller
{
    public function edit()
    {
        $setting = Setting::first();
        return view('admin.settings.setting')->with('setting', $setting);
    }
    public function update()
    {
        $this->validate(request(), [
            'site_name' => 'required',
            'contact_number' => 'required',
            'contact_email' => 'required',
            'address' => 'required',
        ]);
        $setting = Setting::first();
        $setting->site_name = request()->site_name;
        $setting->contact_number = request()->contact_number;
        $setting->contact_email = request()->contact_email;
        $setting->address = request()->address;
        $setting->save();
        session()->flash('success', "Setting updated successfully");
        return redirect(route('dashboard'));
    }
}