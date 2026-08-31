<?php

namespace App\Http\Controllers;

use App\Models\Post;
use Illuminate\Http\Request;

class PostController extends Controller
{
    /**
     * Display a listing of the resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function index()
    {

    }

    /**
     * Show the form for creating a new resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function create()
    {
        //
        return view('admin/add', [
            'title' => 'ADD | MUMUSIK'
        ]);
    }

    /**
     * Store a newly created resource in storage.
     *
     * @param  \Illuminate\Http\Request  $request
     * @return \Illuminate\Http\Response
     */
    public function store(Request $request)
    {
        //dd($request);
        //
        $request->validate([
            'music_name' => 'required',
            'music_album' => 'required',
            'music_author' => 'required',
            'filename'=>'required',
            'filename.*'=> 'file|mimes:audio/mpeg,mpga,mp3,wav,aac']);


            // /storage/upload/files/audio/'.$filename
        $audioName = time().'.'.$request->filename->extension();
        $request->filename->move(storage_path().'audio/', $audioName);
        Post::create([
            'music_name' => $request['music_name'],
            'music_album' => $request['music_album'],
            'music_author' => $request['music_author'],
            'filename'=>$request['filename'],
            'filename'=>$audioName]);

        return redirect('admin/home')->with('success', 'Music has been added.');
    }

    /**
     * Display the specified resource.
     *
     * @param  \App\Models\Post  $post
     * @return \Illuminate\Http\Response
     */
    public function show(Post $post)
    {
        //
    }

    /**
     * Show the form for editing the specified resource.
     *
     * @param  \App\Models\Post  $post
     * @return \Illuminate\Http\Response
     */
    public function edit($post)
    {
        //
        return view('admin/add', [
            'title' => 'UPDATE | MUMUSIK',
            'post' => Post::findorfail($post)
        ]);
    }

    /**
     * Update the specified resource in storage.
     *
     * @param  \Illuminate\Http\Request  $request
     * @param  \App\Models\Post  $post
     * @return \Illuminate\Http\Response
     */
    public function update(Request $request, Post $post)
    {
        //
        //dd($request);
        $post = Post::find($post);
        // /storage/upload/files/audio/'.$filename
        if($request->file('filename')) {
            $audioName = time().'.'.$request->filename->extension();
            $request->filename->move(storage_path().'audio/', $audioName);
        } else {
            $audioName = $post->filename;
        }

        $request->validate([
            'music_name' => 'required',
            'music_album' => 'required',
            'music_author' => 'required',
            'filename'=>'required',
            'filename.*'=> 'file|mimes:audio/mpeg,mpga,mp3,wav,aac']);

            Post::find($post)->update([
                'music_name' => $request['music_name'],
                'music_album' => $request['music_album'],
                'music_author' => $request['music_author'],
                'filename'=>$request['filename'],
                'filename'=>$audioName]);

            return redirect('admin/home')->with('success', 'Music has been Update.');
    }

    /**
     * Remove the specified resource from storage.
     *
     * @param  \App\Models\Post  $post
     * @return \Illuminate\Http\Response
     */
    public function destroy($post)
    {
        //
        $post = Post::find($post);
        $post -> delete();

        return redirect('admin/home')->with('success', 'Music has been Delete.');
    }
}