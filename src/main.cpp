#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
	return EXIT_SUCCESS;
}

/* I HAVE DECIDED TO USE GOOGLE TEST..
//--- Hello, World! for CppUnit

#include <iostream>

#include <cppunit/TestRunner.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/XmlOutputter.h>

class Test : public CPPUNIT_NS::TestCase
{
	CPPUNIT_TEST_SUITE(Test);
	CPPUNIT_TEST(testHelloWorld);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp(void) {}
	void tearDown(void) {} 

protected:
	void testHelloWorld(void) { std::cout << "Hello, world!" << std::endl; }
};

CPPUNIT_TEST_SUITE_REGISTRATION(Test);

int main( int ac, char **av )
{
	//--- Create the event manager and test controller
	CPPUNIT_NS::TestResult controller;

	//--- Add a listener that colllects test result
	CPPUNIT_NS::TestResultCollector result;
	controller.addListener( &result );        

	//--- Add a listener that print dots as test run.
	CPPUNIT_NS::BriefTestProgressListener progress;
	controller.addListener( &progress );      

	//--- Add the top suite to the test runner
	CPPUNIT_NS::TestRunner runner;
	runner.addTest( CPPUNIT_NS::TestFactoryRegistry::getRegistry().makeTest() );
	runner.run( controller );

	std::ofstream xmlFileOut("cppunit-reports/cpptestresults.xml");
	CPPUNIT_NS::XmlOutputter xmlOut(&result, xmlFileOut);
	xmlOut.write();

	return result.wasSuccessful() ? 0 : 1;
}
*/