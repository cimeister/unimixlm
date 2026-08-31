<?php
namespace shop\useCases\auth;

use shop\forms\auth\PasswordResetRequestForm;
use shop\forms\auth\ResetPasswordForm;
use shop\repositories\UserRepository;

class PasswordResetService
{
    private $users;


    public function __construct(UserRepository $users)
    {
       $this->users = $users;
    }

    public function request(PasswordResetRequestForm $form)
    {
        $user = $this->users->getUserByEmail($form->email);

        if (!$user->isActive()){
            throw new \DomainException('User is not active.');
        }
        $user->requestPasswordReset();
        $this->users->save($user);

    }


    public function validateToken($token)
    {
        if (empty($token) || !is_string($token)) {
            throw new \DomainException('Password reset token cannot be blank.');
        }
        if (!$this->users->existByPasswordResetToken($token)) {
            throw new \DomainException('Wrong password reset token.');
        }
    }


    public function reset(string $token, ResetPasswordForm $form)
    {
        $user = $this->users->getUserByPasswordResetToken($token);
        $user->resetPassword($form->password);
        $this->users->save($user);
    }



}