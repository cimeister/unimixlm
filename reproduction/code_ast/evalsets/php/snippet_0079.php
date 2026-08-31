<?php

use App\User;
use Illuminate\Database\Seeder;

class UserSeeder extends Seeder
{
    /**
     * Run the database seeds.
     *
     * @return void
     * @throws Exception
     */
    public function run()
    {
        for($i = 1; $i < 100; $i++) {
            $user = new User();
            $user->name = "Test" . $i;
            $user->password = <PASSWORD>("<PASSWORD>");
            $user->email = "test" . $i . "@test.com";
            $user->save();


            $newAuthToken = hash("sha512", random_bytes(128));
            $token = new \App\AuthTokens();
            $token->user_id = $i;
            $token->token = $newAuthToken;
            $token->save();

        }
    }
}