# Adobe Experience Manager (AEM) Basics

Adobe Experience Manager, usually shortened to AEM, is Adobe's enterprise
content management system for building websites, mobile apps and forms. It
is part of the Adobe Experience Cloud and is widely used by large
organisations to manage digital experiences across many sites and languages
from a single platform.

AEM is built on three open technologies from the Apache Software Foundation.
Apache Sling provides the web framework that maps repository content to URLs
and resolves each request to a script or servlet. The Java Content Repository,
implemented by Apache Jackrabbit Oak, stores all content and assets in a
hierarchical tree. OSGi, through Apache Felix, gives AEM a modular runtime
where features are packaged as bundles that can be installed and updated
independently.

Developers build pages from components, which are reusable building blocks
such as text blocks, images, carousels and teasers. Page templates define
which components authors may place on a page, and editable templates let
authors adjust the layout in the browser. Server-side rendering of
components commonly uses HTL, the HTML Template Language, together with
Sling Models, which are Java classes that expose repository data to the
template in a clean, testable way.

AEM has two main run modes. The author instance is where editors create and
preview content, and the publish instance is the one that serves live
traffic. A replication step copies approved content from author to publish,
and a dispatcher, typically an Apache HTTP server module, caches published
pages and protects the publish instances from unnecessary load.
