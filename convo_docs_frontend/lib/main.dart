import 'package:flutter/material.dart';
import 'package:convo_docs_frontend/features/splash/presentation/screens/splash_screens.dart';
import 'package:convo_docs_frontend/features/splash/presentation/screens/main_layout.dart';


void main() {
  runApp(const ConvoDocsApp());
}

class ConvoDocsApp extends StatelessWidget {
  const ConvoDocsApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BlackNBlue',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      initialRoute: '/',
      routes: {
        '/': (_) => const SplashScreen(),
        '/main_layout': (_) =>  const MainLayout(),
        },

    );

  }
}