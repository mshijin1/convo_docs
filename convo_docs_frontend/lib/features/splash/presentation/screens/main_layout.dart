import 'package:flutter/material.dart';
import 'package:convo_docs_frontend/features/splash/presentation/widgets/sidebar.dart';
void main() {
  runApp(const MainLayout());
}

class MainLayout extends StatelessWidget {
  const MainLayout ({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(

        // AppBar with title
        appBar: AppBar(
          title: const Text('Main Layout'),
        ),

        // Body with a simple welcome message
        body: const Sidebar(),
      ),
    );
  }
}
