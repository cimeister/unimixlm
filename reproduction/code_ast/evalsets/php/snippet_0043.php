<?php

namespace App\Http\Controllers;
use App\Reguler;
use DB;
use Illuminate\Http\Request;
require_once app_path('Classes\PHPEXCEL.php');
require_once app_path('Classes\PHPEXCEL\IOFactory.php');

class exportController extends Controller
{
    public function excel(){

        
        
        $objPHPExcel = new \PHPExcel();
        $objReader = \PHPExcel_IOFactory::createReader('Excel2007');
        $objPHPExcel = $objReader->load(public_path('excel/perso.xlsx'));
        $data = Reguler::with('user', 'pol')->join('proses', 'reguler.id_proses', 'proses.id_proses')->get();
        $no = 1;
        foreach($data as $item){

            $newSheet = clone $objPHPExcel->getActiveSheet();
            $newSheet->setTitle($item->user->username);
            $objPHPExcel->addSheet($newSheet);

            $operator = $item->user->id;
            $id_pol = $item->pol->id_pol;
            $id_proses = $item->id_proses;
            $nomor = 1;
            $row = 12;          
            $q = Reguler::with('pol','user')->join('proses', 'reguler.id_proses', 'proses.id_proses')->where('operator', $operator)->where('id_pol', $id_pol)->where('reguler.id_proses', $id_proses)->get();        
            foreach ($q as $c) {                
                $input = $c->isi + $c->dead + $c->chip_error + $c->chip_lemah;
                $tanggal = tanggal($c->created_at);
                $objPHPExcel->getSheet($no)
                    ->setCellValue('d5', $item->user->username)
                    ->setCellValue('d6', $item->pol->kode_pol)
                    ->setCellValue('d7', $item->pol->jumlah_order)
                    ->setCellValue('j6', $c->nama_proses)
                    // ->setCellValue('s4', $c['group_laporan'])
                    ->setCellValue('b12', $tanggal)
                    ->setCellValue('u3', $c->line)

                    ->setCellValue('a' . $row, $nomor)
                    ->setCellValue('d' . $row, $c->iner)
                    ->setCellValue('e' . $row, jam($c->created_at))
                    ->setCellValue('f' . $row, jam($c->updated_at))
                    ->setCellValue('p' . $row, $c->user->username)
                    ->setCellValue('h' . $row, $c->isi)
                    ->setCellValue('i' . $row, $c->dead)
                    ->setCellValue('o' . $row, $c->chip_error)
                    ->setCellValue('j' . $row, $c->chip_lemah)
                    ->setCellValue('g' . $row, $input);
                $row++;
                $nomor++;
            }
            $no++;
        }
        $objPHPExcel->removeSheetbyindex(0);

        header('Content-Type: application/vnd.ms-excel');
        header('Content-Disposition: attachment;filename=Laporan Harian Perso s.xlsx');
        header('Cache-Control: max-age=0');

        $objWriter = \PHPExcel_IOFactory::createWriter($objPHPExcel, 'Excel2007');
        $objWriter->save('php://output');
        exit;
    }
}