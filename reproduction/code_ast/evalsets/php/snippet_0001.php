<?php

namespace MarkusGehrig\Diary\Mail;

// Copyright (c) 2018 <NAME>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use \Swift_SmtpTransport;
use \Swift_Message;
use \Swift_Mailer;

class MailSender
{
    private $config = [];
    private $transport;
    private $mailer;

    public function __construct()
    {
        $this->config = $this->getSmtpCredentialsFromConfiguration();

        //var_dump($this->config);

        $this->transport = $this->getTransport();
        $this->mailer = new Swift_Mailer($this->transport);
    }

    public function createMail($subject, $sender, $receiver, $body)
    {
        return (new Swift_Message($subject))
            ->setFrom((array) $sender)
            ->setTo((array) $receiver)
            ->setBody($body)
            ->setContentType("text/html");
        ;
    }

    public function send($message)
    {
        $result = $this->mailer->send($message);
        print_r($result);
    }

    private function getSmtpCredentialsFromConfiguration()
    {
        return $GLOBALS['configuration']['mail'];
    }

    private function getTransport()
    {
        return (new Swift_SmtpTransport($this->config['server'], $this->config['port'], 'tls'))
            ->setUsername($this->config['user'])
            ->setPassword($this->config['password']);
    }
}