<?php

namespace Seller\Controller;

use Zend\Mvc\Controller\AbstractActionController;
use Zend\View\Model\ViewModel;
use Zend\Http\PhpEnvironment\Request;
use Seller\Model\Seller;
use Seller\Form\SellerForm;
use Seller\Form\UploadForm;
use Seller\Form\EmailForm;
use Zend\Mail\Message;
use Zend\Mail\Transport\Sendmail;
use Zend\Mail;
use Zend\Db\Sql\Select;
use Zend\Paginator\Paginator;
use Zend\Paginator\Adapter\Iterator as paginatorIterator;

class SellerController extends AbstractActionController {

    public function indexAction() {
        
        if ($this->zfcUserAuthentication()->hasIdentity()) {
    //get the email of the user
        $user_id= $this->zfcUserAuthentication()->getIdentity()->getId();
        
        }
        
        $form = new UploadForm('upload-form');

        if ($this->getRequest()->isPost()) {
            // Make certain to merge the files info!
            $post = array_merge_recursive(
                    $this->getRequest()->getPost()->toArray(), $this->getRequest()->getFiles()->toArray()
            );

            $form->setData($post);

            $seller = new Seller();
            //$form->setInputFilter($seller->getInputFilter());


            if ($form->isValid()) {
                $data = $form->getData();

                $property_image = substr(($data['image-file']['tmp_name']), -24);

                $data['property_image'] = $property_image;
                
                $data['user_id'] = $user_id;

                $seller = new Seller();

                $seller->exchangeArray($data);
                $this->getSellerTable()->saveProperty($seller);


                // Form is valid, save the form!
                return $this->redirect()->toRoute('seller');
            }
        }

        return array('form' => $form);
    }

    public function getSellerTable() {
//        if (!$this->sellerTable) {
        $sm = $this->getServiceLocator();
        $this->sellerTable = $sm->get('Seller\Model\SellerTable');
//        }
        return $this->sellerTable;
    }

    /*     * ************************************************* */

    public function uploadFormAction() {
        
       
        $form = new UploadForm('upload-form');
        $tempFile = null;

        $prg = $this->fileprg($form);
        if ($prg instanceof \Zend\Http\PhpEnvironment\Response) {
            return $prg; // Return PRG redirect response
        } elseif (is_array($prg)) {
            if ($form->isValid()) {
                $data = $form->getData();
                // Form is valid, save the form!
                return $this->redirect()->toRoute('upload-form/success');
            } else {
                // Form not valid, but file uploads might be valid...
                // Get the temporary file information to show the user in the view
                $fileErrors = $form->get('image-file')->getMessages();
                if (empty($fileErrors)) {
                    $tempFile = $form->get('image-file')->getValue();
                }
            }
        }

        return array(
            'form' => $form,
            'tempFile' => $tempFile,
        );
    }

    public function detailsAction() {


        $id = (int) $this->params('id');

        if ($id != 0) {
            $property_details = $this->getSellerTable()->getProperty($id);
        }



        $emailForm = new EmailForm('email-form');
        $emailForm->bind($property_details);

        $request = $this->getRequest();

        if ($this->getRequest()->isPost()) {

            $data = $request->getPost();

            $email = $data->user_email;

            /*
              $message = new Message();

              $message->addFrom($email)
              ->addTo("<EMAIL>")
              ->setSubject("Sending an email from Property Details Page");
              $message->setBody("This is the message body.");

              $transport = new Mail\Transport\Sendmail();
              $transport->send($message);
             */



//            $mail = new Mail\Message();
//            $mail->setBody('This is the text of the email.');
//            $mail->setFrom($email);
//            $mail->addTo('<EMAIL>', 'Iqbal');
//            $mail->setSubject('TestSubject');
//
//            $transport = new Mail\Transport\Sendmail('-f<EMAIL>');
//            $transport->send($mail);


            return $this->redirect()->toRoute('application');
        }

        return array('form' => $emailForm, 'property_details' => $property_details);
    }

    public function allpropertyAction() {
        
         $select = new Select();

        $order_by = $this->params()->fromRoute('order_by') ?
                $this->params()->fromRoute('order_by') : 'id';
        $order = $this->params()->fromRoute('order') ?
                $this->params()->fromRoute('order') : Select::ORDER_ASCENDING;
        $page = $this->params()->fromRoute('page') ? (int) $this->params()->fromRoute('page') : 1;

        $albums = $this->getSellerTable()->fetchAllProperty($select->order($order_by . ' ' . $order));
        $itemsPerPage = 1;

        $albums->current();
        $paginator = new Paginator(new paginatorIterator($albums));
        $paginator->setCurrentPageNumber($page)
                ->setItemCountPerPage($itemsPerPage)
                ->setPageRange(7);

        return new ViewModel(array(
                    'order_by' => $order_by,
                    'order' => $order,
                    'page' => $page,
                    'paginator' => $paginator,
                ));
        
        
        
    }

    public function contactAction(){
        
        $contactForm = new \Seller\Form\ContactForm();
        
         $request = $this->getRequest();

        if ($this->getRequest()->isPost()) {

            $data = $request->getPost();

            $user_name = $data->user_name;
            
            $user_phone = $data->user_phone;
            
           
           $email = $data->user_email;
           
           
            
           $user_message = 'Phone:'.$user_phone."\r\n"; 
           
          
           $user_message .= $data->user_message;
                 
           

            $mail = new Mail\Message();
            $mail->setBody($user_message);
           
            $mail->setFrom($email,$user_name);
            $mail->addTo('<EMAIL>', 'Iqbal');
            $mail->setSubject('TestSubject');

            $transport = new Mail\Transport\Sendmail('-f<EMAIL>');
        $transport->send($mail);
        
        
        }
        
        return array('form'=>$contactForm);
    } 
    
        
        
    
    
    
    public function aboutUsAction(){
        
        
    }
    
    
    public function faqAction() {
        
    }
    
    
}