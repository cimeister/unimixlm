<?php

namespace PlanIt\RestBundle\Controller;

use Symfony\Bundle\FrameworkBundle\Controller\Controller;
use Symfony\Component\HttpFoundation\Request;
use PlanIt\EventBundle\Entity\Event;
use PlanIt\GuestsBundle\Entity\GuestsModule;
use PlanIt\BudgetBundle\Entity\BudgetModule;
use PlanIt\PlaceBundle\Entity\PlaceModule;
use PlanIt\TransportationBundle\Entity\TransportationModule;
use PlanIt\TodoBundle\Entity\TodoModule;
use PlanIt\EventBundle\Form\EventType;
use Symfony\Component\DependencyInjection\ContainerInterface;


class EventRestController extends Controller
{

    protected $container;
    public function __construct(ContainerInterface $container = null)
    {
        $this->container = $container;
    }


    public function getEventNotusemodulesAction($event_id){
        $event = $this->getDoctrine()->getRepository('PlanItEventBundle:Event')->find($event_id);
        $modules_tab = array(
            1 => "Gestion des invités",
            2 => "Gestion du budget",
            3 => "Gestion du lieu",
            4 => "Gestion du transport",
            5 => "Liste de tâches"
        );

           foreach($modules_tab as $type => $name){
                foreach($event->getModules() as $module){
                    if($type == $module->getIntType()){
                        unset($modules_tab[$type]);
                    }
               }
           }
           $notUseModules = [];
           foreach($modules_tab as $type=>$name){
                if($type == 1){
                    $module = new GuestsModule();
                    $module->setIntType(1);
                    $module->setName('Gestion des invités');
                }elseif($type == 2){
                    $module = new BudgetModule();
                    $module->setIntType(2);
                    $module->setName('Gestion du budget');
                }elseif($type == 3){
                    $module = new PlaceModule();
                    $module->setIntType(3);
                    $module->setName('Gestion du lieu');
                }elseif($type == 4){
                    $module = new TransportationModule();
                    $module->setIntType(4);
                    $module->setName('Gestion du transport');
                }elseif($type == 5){
                    $module = new TodoModule();
                    $module->setIntType(5);
                    $module->setName('Listes de tâches');
                }
                $notUseModules[] = $module;
           }
           return $notUseModules;
    }
    
    public function getEventAction($event_id){
	    $event = $this->getDoctrine()->getRepository('PlanItEventBundle:Event')->find($event_id);
        $nbGuests = $this->getEventNbguestsAction($event_id);
	    if(!is_object($event)){
	      throw $this->createNotFoundException();
	    }
        $balance = "Empty";
        $total_expenses = 0;
        $total_inflows = 0;
        $guests_inflow = 0;
        foreach ($event->getModules() as $module) {
            if ($module->getIntType() == 2){
                $total_expenses = $module->getTotalExpenses();
                $total_inflows = $module->getTotalInflows();
                $balance = $module->getBalance();
                $guests_inflow = $this->get("budget_api_controller")->getGuestsinflowAction($module->getId());
                $balance += $guests_inflow;
            }            
        }
        $modules = array();
        foreach($event->getModules() as $module){
            $modules[] = array(
                'id' => $module->getId(),
                'name' => $module->getName(),
                'inttype' => $module->getInttype()
            );
        }

	    return array(
                    'nbGuests' => $nbGuests,
                    'event' => array(
                        'id' => $event->getId(),
                        'name' => $event->getName(),
                        'description' => $event->getDescription(),
                        'beginDate' => $event->getBeginDate(),
                        'endDate' => $event->getEndDate(),
                        'image' => $event->getImage(),
                        'modules' => $modules,
                        'user' => $event->getUser()
                    ),
                    'balance' => $balance,
                    'guests_inflow' => $guests_inflow,
                    'total_expenses' => $total_expenses,
                    'total_inflows' => $total_inflows
                );
	}



	public function postEventAction(Request $request, $user_id)
    {
        $user = $this->getDoctrine()->getRepository('PlanItUserBundle:User')->find($user_id);  
        $event = new Event();
        $event->setUser($user);
        if($request->request->get('event_form') == null){
            if($request->request->get('begin_date') <= $request->request->get('end_date')){
                $event->setName($request->request->get('name'));
                $event->setSlug($request->request->get('name'));
                $event->setDescription($request->request->get('description'));
                $event->setBeginDate(\DateTime::createFromFormat('j/m/Y', $request->request->get('begin_date')));
                $event->setEndDate(\DateTime::createFromFormat('j/m/Y', $request->request->get('end_date')));
                if($request->files->get('image') != null){
                    $file = $request->files->get('image');
                    $extension = $file->guessExtension();
                    if (!$extension) {
                        $extension = 'bin';
                    }

                    $rand = rand(1, 99999);
                    $file->move($event->getUploadRootDir(), $user_id.'-'.$rand.'.'.$extension);
                    $event->setImage($user_id.'-'.$rand.'.'.$extension);
                }
                $em = $this->getDoctrine()
                               ->getEntityManager();
                    $em->persist($event);
                    $em->flush();
                return 'ok';
            }else{
                return 'La date de fin est inférieure à la date de début';
            }
            
        }else{
            $form    = $this->createForm(new EventType("add"), $event);
            $form->handleRequest($request);
            $session = $request->getSession();
            // if(!empty($form['end_date']->getData()) && $form['begin_date']->getData() >= $form['end_date']->getData()){
            //     $session->getFlashBag()->add('errors', 'Attention, la date de fin doit être postérieure à la date de début');
            //     return $this->redirect($this->generateUrl('PlanItUserBundle_homepage', array(
            //             'id'    => $user_id
            //         )));
            // }
            if ($form->isValid()) {
                $data = $form->getData();
                // if(empty($form['end_date']->getData())){
                //     $event->setEndDate($form['begin_date']->getData());
                // }
                $event->setSlug($data->getName());

                if($form['image']->getData() != null){
                    $file = $form['image']->getData();
                    $extension = $file->guessExtension();
                    if (!$extension) {
                        $extension = 'bin';
                    }

                    $rand = rand(1, 99999);
                    $file->move($event->getUploadRootDir(), $user_id.'-'.$rand.'.'.$extension);
                    $event->setImage($user_id.'-'.$rand.'.'.$extension);
                }
                $em = $this->getDoctrine()
                           ->getEntityManager();
                $em->persist($event);
                $em->flush();

               return $this->redirect($this->generateUrl('PlanItEventBundle_event', array(
                    'id'    => $event->getId()
                )));
            }
            $errors = $this->get('validator')->validate( $event );
            foreach( $errors as $error )
            {
                $session->getFlashBag()->add('errors', $error->getMessage());
            }
            return $this->redirect($this->generateUrl('PlanItUserBundle_homepage', array(
                        'id'    => $user_id
                    )));
        }
    }

    public function deleteModuleAction($module_id)
    {
        
        $module = $this->getDoctrine()->getRepository('PlanItModuleBundle:Module')->find($module_id);
        $event = $module->getEvent();
        $nbGuests = $this->getEventNbguestsAction($event->getId());
        $em = $this->getDoctrine()
                       ->getEntityManager();
        switch($module->getIntType()){
            case 1: //guests
                $types_guests = $module->getTypesGuests();
                foreach ($types_guests as $type) {
                    $guests = $type->getGuests();
                    foreach ($guests as $guest) {
                        $em->remove($guest);
                        $em->flush();
                    }
                    $em->remove($type);
                    $em->flush();
                }
                $payments = $module->getPaymentMeans();
                foreach ($payments as $payment) {
                    $em->remove($payment);
                    $em->flush();
                }
                //return 'ok'
            break;

            case 2: //budget
                $types_expense = $module->getTypesExpense();
                foreach ($types_expense as $type) {
                    $expenses = $type->getExpenses();
                    foreach ($expenses as $expense) {
                        $em->remove($expense);
                        $em->flush();
                    }
                    $em->remove($type);
                    $em->flush();
                }
                $inflows = $module->getInflows();
                foreach ($inflows as $inflow) {
                    $em->remove($inflow);
                    $em->flush();
                }

            break;


            case 3: //place
                $places = $module->getPlaces();
                foreach ($places as $place) {
                    $em->remove($place);
                    $em->flush();
                }
            break;

            case 4:
                $transportations = $module->getTransportations();
                foreach ($transportations as $transportation) {
                    $em->remove($transportation);
                    $em->flush();
                }

            break;

            case 5: //todo
                $lists = $module->getLists();
                foreach ($lists as $list) {
                    $items = $list->getItems();
                    foreach ($items as $item) {
                        $em->remove($item);
                        $em->flush();
                    }

                    $em->remove($list);
                    $em->flush();
                }
            break;


        }
        $em->remove($module);
        $em->flush();
        return $this->getEventNotusemodulesAction($event->getId());

    }

    public function deleteEventAction($event_id)
    {
        $event = $this->getDoctrine()->getRepository('PlanItEventBundle:Event')->find($event_id);
        $em = $em = $this->getDoctrine()
                       ->getEntityManager();
        $em->remove($event);
        $em->flush();
    }

    public function postEventUpdateAction(Request $request, $event_id)
    {
        
        $event = $this->getDoctrine()->getRepository('PlanItEventBundle:Event')->find($event_id);
        $type = "update";
        $form    = $this->createForm(new EventType($type), $event);
        $form->handleRequest($request);
        if ($form->isValid()) {
            $data = $form->getData();
            $em = $this->getDoctrine()
                       ->getEntityManager();
            $em->persist($event);
            $em->flush();
            
            return $this->redirect($this->generateUrl('PlanItEventBundle_event', array(
                'id'    => $event->getId()
            )));
        }

        return $this->redirect($this->generateUrl('PlanItEventBundle_event', array(
                'id'    => $event->getId()
            )));
    }

    public function getEventNbguestsAction($event_id){
        return $nbGuests = $this->getDoctrine()->getRepository('PlanItGuestsBundle:Guest')->countGuests($event_id);
    }

}