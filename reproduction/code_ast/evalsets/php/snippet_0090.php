<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Validator;
use App\Models\User;
use App\Models\Item;
use App\Models\Edital;

class EditalController extends Controller
{
    /**
     * Create a new controller instance.
     *
     * @return void
     */
    public function __construct()
    {
        $this->middleware('auth');
    }
    /**
     * Display a listing of the resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function index()
    {
        $itens = Item::all();
        return view('painel.edital', ['itens' => $itens]);
    }

    public function items()
    {
        $itens = Item::all();

        return view('painel.items', [
            'itens' => $itens
        ]);
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function create()
    {
    }

    /**
     * Store a newly created resource in storage.
     *
     * @param  \Illuminate\Http\Request  $request
     * @return \Illuminate\Http\Response
     */
    public function store(Request $request)
    {
        $validator = Validator::make(
            $request->all(),
            [
            'tipo'      => ['required', 'string', 'max:10', 'min:1'],
            'ano'     => ['required','string', 'max:50'],
            'tipo'      => ['required'],
            'jom'       => ['required','mimes:pdf'],
            'path_pdf'     => ['required','mimes:pdf']
            ]
        );

        if ($validator->fails()) {
            return redirect('/register_edital')
                        ->withErrors($validator)
                        ->withInput();
        }

        $user = Auth::user();
        $tipo = $request->input('tipo');
        $orgao = $request->input('orgao');
        $ano = $request->input('ano');

        $edital = new Edital();
        $edital->user_id = $user->id;
        $edital->tipo = $tipo;
        $edital->orgao = $orgao;
        $edital->ano = $ano;
        $edital->path_pdf = $request->file('path_pdf')->store('pdfs');
        $edital->jom = $request->file('jom')->store('jom');
        if ($edital->save()) {
                return redirect()->route('home')->with("success", "Salvo com sucesso");
        } else {
            return redirect()->route('home')->with("warning", "Erro ao salvar");
        }
    }

    /**
     * Display the specified resource.
     *
     * @param  int  $id
     * @return \Illuminate\Http\Response
     */
    public function show($id)
    {
        return view('painel.invoice');
    }

    /**
     * Show the form for editing the specified resource.
     *
     * @param  int  $id
     * @return \Illuminate\Http\Response
     */
    public function edit($id)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     *
     * @param  \Illuminate\Http\Request  $request
     * @param  int  $id
     * @return \Illuminate\Http\Response
     */
    public function update(Request $request)
    {
        $id = $request->input('id');
        Item::findOrFail($id)->update($request->only(['condicao']));
        return redirect()->route('home')->with("editItem", "Salvo com sucesso");
    }

    /**
     * Remove the specified resource from storage.
     *
     * @param  int  $id
     * @return \Illuminate\Http\Response
     */
    public function destroy($id)
    {
        //
    }
}