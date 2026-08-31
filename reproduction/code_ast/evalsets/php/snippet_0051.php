<?php
namespace Slaleye\Memory\Tests\Unit\Domain\Model;

/**
 * Test case.
 *
 * @author Slaleye 
 */
class HighscoreTest extends \TYPO3\CMS\Core\Tests\UnitTestCase
{
    /**
     * @var \Slaleye\Memory\Domain\Model\Highscore
     */
    protected $subject = null;

    protected function setUp()
    {
        parent::setUp();
        $this->subject = new \Slaleye\Memory\Domain\Model\Highscore();
    }

    protected function tearDown()
    {
        parent::tearDown();
    }

    /**
     * @test
     */
    public function getUsernameReturnsInitialValueForString()
    {
        self::assertSame(
            '',
            $this->subject->getUsername()
        );
    }

    /**
     * @test
     */
    public function setUsernameForStringSetsUsername()
    {
        $this->subject->setUsername('Conceived at T3CON10');

        self::assertAttributeEquals(
            'Conceived at T3CON10',
            'username',
            $this->subject
        );
    }

    /**
     * @test
     */
    public function getScoreReturnsInitialValueForInt()
    {
        self::assertSame(
            0,
            $this->subject->getScore()
        );
    }

    /**
     * @test
     */
    public function setScoreForIntSetsScore()
    {
        $this->subject->setScore(12);

        self::assertAttributeEquals(
            12,
            'score',
            $this->subject
        );
    }
}