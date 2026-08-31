<?php

namespace Nemundo\Admin\Com\Navigation;


use Nemundo\Package\Bootstrap\Tabs\BootstrapSiteTabs;

// AdminSiteTabs
class AdminNavigation extends BootstrapSiteTabs
{

    public function getContent()
    {

        $this->addCssClass('mt-3');
        $this->addCssClass('mb-3');

        return parent::getContent();
    }

}