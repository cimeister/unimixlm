<?php

namespace App\Http\Controllers;
use App\Blog;
use Illuminate\Http\Request;

class AdminController extends Controller
{
    public function __construct()
    {
        $this->middleware(['auth', 'admin'], ['only' => 'blogs']);
        $this->middleware('auth');
    }


    public function index(){
        return view('admin.dashboard');
    }

    public function blogs(){
        $published_blogs = Blog::where('status', 1)->get();
        $draft_blogs = Blog::where('status', 0)->get();
        return view('admin.blogs')->with(['published_blogs' => $published_blogs, 'draft_blogs' => $draft_blogs]);
    }
}