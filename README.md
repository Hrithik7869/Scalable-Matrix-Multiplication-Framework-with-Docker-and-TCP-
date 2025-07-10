Scalable Matrix Multiplication Framework with Docker and TCP
Overview
This project, developed as part of our Graduate Systems semester coursework, presents a scalable and modular framework for distributed matrix multiplication. Using a containerized architecture built with Docker and a simple TCP-based communication model, the system demonstrates an effective way to parallelize heavy computational tasks across multiple worker nodes.
Matrix multiplication is a fundamental operation in numerous scientific, engineering, and machine learning applications. However, it can become computationally expensive as matrix sizes grow. Distributing the workload across several compute nodes or services can drastically reduce computation time and network bottlenecks.
Our framework simulates a distributed environment by leveraging Docker containers as independent workers, each responsible for computing a specific portion of the final output matrix. A central container orchestrates task distribution and result aggregation. This setup helps students and researchers understand the concepts of parallelization, distributed systems, container networking, and basic service orchestration.
________________________________________
Motivation
Modern distributed systems rely heavily on efficient data partitioning and task distribution to achieve scalability and high throughput. In large-scale data centers and high-performance computing (HPC) environments, operations like matrix multiplication are often offloaded to specialized hardware or distributed across clusters to reduce execution time.
The motivation for this project arose from the need to explore these techniques in an accessible, controlled, and educational setting. By using Docker to emulate a distributed environment, we were able to avoid the complexity of physical clusters while still providing realistic, network-based communication between services. Furthermore, using C for computation ensures that students engage with low-level concepts such as socket programming, data serialization, and manual memory management.
________________________________________
Architecture
The framework is divided into a central orchestrator service and four worker services, each running inside its own Docker container. The architecture can be summarized as follows:
•	Central container:
o	Reads two 4×4 matrices (A and B).
o	Divides matrix A row-wise and sends each row, along with the complete matrix B, to a different worker.
o	Receives computed result rows from each worker and assembles the final output matrix C.
o	Displays or stores the final matrix.
•	Worker containers:
o	Each runs a TCP server implemented in C.
o	Waits for connection from the central container.
o	Receives one row from matrix A and the full matrix B.
o	Performs matrix multiplication for its assigned row.
o	Sends the computed row back to the central container.
•	Docker networking:
o	All containers are connected through a custom bridge network called matrix_net.
o	This facilitates smooth TCP communication between the central and worker services.
________________________________________
Technical Details
Docker Compose
The system uses a docker-compose.yml file to define and orchestrate all services:
•	Defines one central service and four worker services.
•	Each worker service is configured to run on a specific mapped port (5101 to 5104) to receive connections from the central service.
•	Containers are built using custom Dockerfiles (Dockerfile.central for the central service and Dockerfile.worker for the workers).
•	The bridge network (matrix_net) ensures that containers can discover and connect to each other using container names.
________________________________________
Code Implementation
Central Service (central.c)
The central service is responsible for:
•	Initializing matrices.
•	Establishing TCP connections to each worker.
•	Sending a specific row of A and entire B to each worker.
•	Receiving processed rows and assembling the final result.
The code demonstrates:
•	Socket creation and connection.
•	Use of send and recv functions to transmit data between containers.
•	Basic error handling and resource cleanup.
Worker Service (worker.c)
Each worker service:
•	Listens on its designated TCP port.
•	Accepts a connection from the central service.
•	Receives one row and full matrix data.
•	Performs single-row matrix multiplication logic.
•	Sends back the result row.
The code showcases:
•	Server-side socket programming basics in C.
•	Matrix manipulation logic.
•	Clean shutdown and minimal thread usage to keep design simple.
________________________________________
Educational Takeaways
Through this project, we developed a strong practical understanding of several key system concepts:
•	Distributed Computing: Learning how to split computation into smaller independent tasks and coordinate among distributed services.
•	Socket Programming in C: Handling raw data transmission over TCP, managing sockets, and ensuring correct synchronization between client and server.
•	Containerization and Docker: Using Docker to simulate multiple compute nodes, creating custom images, and orchestrating services with Docker Compose.
•	Network Configuration: Understanding Docker bridge networks and internal container communication.
Additionally, we gained valuable experience in troubleshooting inter-container networking issues, ensuring deterministic data flow, and maintaining clean build environments through Dockerfiles.
________________________________________
Challenges Faced
While developing this framework, we encountered and addressed several challenges:
•	Synchronization and Deadlocks: Ensuring that each worker correctly receives its assigned data without race conditions or unexpected hangs.
•	Container Networking: Initially, containers could not resolve each other by name, which we resolved by defining a custom bridge network (matrix_net).
•	Data Serialization: Sending raw integer arrays over TCP required careful handling of buffer sizes and consistent data structures.
•	Port Conflicts and Mapping: Assigning proper host-to-container port mappings to prevent clashes and ensure smooth connection establishment.
By solving these, we not only strengthened our debugging skills but also developed a deeper appreciation for distributed system design principles.
________________________________________
Performance and Future Enhancements
In our setup, even though the matrix size is relatively small (4×4), we demonstrated significant theoretical speedups and practical parallelization. With larger matrices, this approach can be extended to showcase measurable improvements in computation time.
Possible future enhancements include:
•	Dynamic Scaling: Supporting dynamic addition or removal of worker containers based on workload.
•	Thread Pooling: Modifying worker logic to handle multiple tasks concurrently using threads.
•	Advanced Orchestration: Integrating Kubernetes or Docker Swarm for more robust container management and scaling.
•	Generalization: Extending support to arbitrary matrix sizes and improving data partitioning strategies.
________________________________________
Teamwork and Contributions
This project was executed by a team of four MTech students, with each member contributing to different aspects:
•	Container orchestration & networking setup
•	Central and worker C code design and debugging
•	Dockerfile and build environment optimization
•	Testing, integration, and final reporting
The collaborative effort allowed us to explore different roles and work in an environment similar to real-world software engineering teams.
________________________________________
Conclusion
Our "Scalable Matrix Multiplication Framework with Docker and TCP" demonstrates a practical, modular approach to distributed computation in an easily reproducible environment. By combining low-level C programming with modern containerization, we effectively bridged theoretical system concepts with hands-on implementation.
We believe this framework can serve as a valuable educational tool and foundation for further research or experimentation in distributed systems, parallel computing, and container-based architectures.
________________________________________
How to Use
1.	Clone the repository:
  
 
git clone https://github.com/Hrithik7869/Scalable-Matrix-Multiplication-Framework-with-Docker-and-TCP-.git
cd Scalable-Matrix-Multiplication-Framework-with-Docker-and-TCP-
2.	Build and run the containers:
  
 
docker-compose up --build
3.	View output:
Once all containers start, the central service will display the final matrix multiplication result in your terminal.
4.	Stop services:
  
 
docker-compose down
________________________________________
Final Thoughts
We thoroughly enjoyed working on this project as part of our Graduate Systems course, and it helped us develop not only technical but also collaboration and problem-solving skills. We hope this serves as a helpful reference for anyone interested in learning more about distributed computing and containerized service design.

