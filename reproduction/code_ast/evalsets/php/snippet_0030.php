<?php

namespace AppBundle\Form\Type;

use AppBundle\Repository\GroupeRepository;
use Doctrine\ORM\EntityRepository;
use Symfony\Bridge\Doctrine\Form\Type\EntityType;
use Symfony\Component\Form\AbstractType;
use Symfony\Component\Form\Extension\Core\Type\BirthdayType;
use Symfony\Component\Form\Extension\Core\Type\EmailType;
use Symfony\Component\Form\FormBuilderInterface;
use Symfony\Component\OptionsResolver\OptionsResolver;
use UserBundle\Entity\User;

class IndividuType extends AbstractType
{
    /**
     * {@inheritdoc}
     */
    public function buildForm(FormBuilderInterface $builder, array $options)
    {
        $builder->add('identifiant')
            ->add('nom')
            ->add('prenom')
            ->add('email', EmailType::class)
            ->add('telephone')
            ->add('dateNaissance', BirthdayType::class, array(
                'widget' => 'single_text',
                'html5' => false,
                'format' => 'yyyy-MM-dd'))
            ->add('statut', EntityType::class, array(
                'class' => 'AppBundle\Entity\Statut',
                'choice_label' => 'libelle',
                'multiple' => false,
            ))
            ->add('groupes', EntityType::class, array(
                'class' => 'AppBundle\Entity\Groupe',
                'choice_label' => 'nom',
                'multiple' => true,
                'required' => false,
                'by_reference' => false // Force l'utilisation du setter pour modifier cet attribut (le setter a été modifié pour que les groupes soient bien persistés)
            ))
        ;
    }
    
    /**
     * {@inheritdoc}
     */
    public function configureOptions(OptionsResolver $resolver)
    {
        $resolver->setDefaults(array(
            'data_class' => 'AppBundle\Entity\Individu'
        ));
    }

    /**
     * {@inheritdoc}
     */
    public function getBlockPrefix()
    {
        return 'appbundle_individu';
    }


}