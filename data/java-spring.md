# Java and the Spring Framework

Spring is the most widely used framework for building enterprise Java
applications. At its core is the Inversion of Control container, which
creates objects and wires them together through dependency injection, so
classes declare the collaborators they need instead of constructing them.
This makes code easier to test, because dependencies can be replaced with
mocks or stubs.

Spring Boot builds on Spring to make standalone applications easy to create
and run. Its best-known feature is auto-configuration: Spring Boot inspects
the libraries on the classpath and the properties the developer has set, and
then configures beans automatically — for example, if an embedded database
driver is present, it sets up a data source without manual XML or Java
configuration. Developers can always override these defaults.

A typical Spring Boot web application uses Spring MVC to expose REST
endpoints: classes annotated with @RestController map HTTP requests to
methods, and responses are serialised to JSON automatically. Spring Data JPA
simplifies database access by generating repository implementations from
interface method names. Spring Boot Actuator adds production-ready features
such as health checks, metrics and environment information over HTTP
endpoints, which operations teams use to monitor running applications.
