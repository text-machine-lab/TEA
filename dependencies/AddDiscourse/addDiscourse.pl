## Emily Pitler
## 12/9/2009

## Including bug fix by Richard Johansson 6/13/2011

##Tool to identify discourse connectives
##Given a directory of parse trees, outputs parse trees annotated with discourse connectives
##trained on the Penn Discourse Treebank, based on the features in 
##"Using Syntax to Disambiguate Explicit Discourse Connectives in Text" Pitler&Nenkova ACL 2009

####TODO: Make sure not redoing something already done when computing features (for example, as as in as soon as wsj2300)


#!/usr/bin/perl

use Getopt::Long;
use strict;

my $dir = ""; #directory containing parsed files to analyze for discourse
my $outputDir = ""; #directory to place parsed files with discourse annotations

GetOptions('parses=s'=>\$dir, 'output=s'=>\$outputDir);

if (not $dir) {
  die "Usage: perl addDiscourse.pl --parses [file/directory to parse] --output [file/directory to put annotations]";
}

#read in list of connectives
my %connectives = (); #key is lower-cased known connectives, value is 1
my %longDistConnectives = (); #key is lower-cased first half, value is lower-cased second half
open(FIN, "resources/connectives.txt") or die "Couldn't open connectives.txt";
foreach my $c (<FIN>) {
  chomp($c);
  #only look for first half and then add in second-half in post-processing
  if ($c =~ s/\.\.(.*)//) {
    $longDistConnectives{$c} = $1;
  } 
  $connectives{$c} = 1;
}
close(FIN);

#read in list of connective texts
my %connectiveTexts = (); #key is lower-cased known connectives, value is 1
open(FIN, "resources/connectiveTexts.txt") or die "Couldn't open connectiveTexts.txt";
foreach my $c (<FIN>) {
  chomp($c);
  $connectiveTexts{$c} = 1;
}
close(FIN);

my %labelWeights = (); #hash from labels to features to weights
open(FIN, "resources/connectives.info") or die "Couldn't open classifier.info";
my $currLabel = "";
foreach my $line (<FIN>) {
  chomp($line);
  if ($line =~ /FEATURES FOR CLASS (.*)/) {
    $currLabel = $1;
  }
  else {
    $line =~ s/^\s+//;
    my ($f, $w) = split /\s+/, $line;
    $labelWeights{$currLabel}{$f} = $w;
  }
}
close(FIN);
my @posLabels = keys %labelWeights;

my @files = ();
if (-d $dir) {
  @files = <$dir/*>;
}
else {
  @files = ($dir);
}
  foreach my $file (@files) {
    $file =~ /\/([^\/\.]*)(\.|$)/;
    my $ending = $1;

    ###go through parsed text   
    open(FIN, $file) or die "Couldn't open $file";
    my @lines = ();
    foreach my $line (<FIN>) {
      chomp($line);
      push @lines, $line;
    }
    my $parsedText = join "", @lines;
    my @newParsedLines = @lines;
    my $newParsedText = join "\n", @lines;
    
    my $idInFile = 0;

    #check for everything in connectives
    foreach my $conn (sort {length($b)<=>length($a)} keys %connectives) {
      my $overallId = 0;
      my @words = split / /, $conn;
my $regex = '\\b'.join('\\)*\s+(\s|\\(\\S*|\))*', @words) . '\\)';
      while ($parsedText =~ /$regex/isg) {
        my $match = $&;
        $match =~ s/\)$//; #remove ending parens from match
        my $pos = pos($parsedText)-1; #pos is immediately after end of word
       
        my ($selfCat, $parentCat, $leftSibCat, $rightSibCat, $rightSibWVP, $rightSibWTrace) = ("S","P","L","R","RSVPN","RSTN"); 

        ##Find self-category-highest that includes only match 
        my $parensEndingSelf = 0;
        my $parsedText2 = $parsedText;
        #number of parenthesis around self is min of (open before, close after)
        #find afterCloseBeforeOpen
        my $i = $pos;
        my $afterCloseBeforeOpen = 0;
	my $len = length($parsedText2);
        while (($i < $len) && (substr($parsedText2, $i, 1) ne '(')) {
          if (substr($parsedText2, $i, 1) eq ')') {
            $afterCloseBeforeOpen++;
          }
          $i++;
        }
        #find prevOpenBeforeClose

        ##two cases--normal case where is a constituent ``but'', ``on the other hand''
        my $diff = ($match =~ s/\(/\(/g) - ($match =~ s/\)/\)/g);
        my $i = $pos-length($match);
        if ($diff == 1 or scalar(@words)==1) {
          my $prevOpenBeforeClose = $diff;
          while ($prevOpenBeforeClose < $afterCloseBeforeOpen and not substr($parsedText2, $i, 1) eq ')') {
            if (substr($parsedText2, $i, 1) eq '(') {
              $prevOpenBeforeClose++;
            }
            $i--;
          }
          $parensEndingSelf = $prevOpenBeforeClose;
        }

        ##pathological case where not a constituent ``as soon as'' (take last part as the head)
        else {
          $i = $pos-1;
          my $prevOpenBeforeClose = 0;
          while ($prevOpenBeforeClose < $afterCloseBeforeOpen) {
            if (substr($parsedText2, $i, 1) eq '(') {
              $prevOpenBeforeClose++;
            }
            if (substr($parsedText2, $i, 1) eq ')') {
              $prevOpenBeforeClose--;
            }
            $i--;
          }
          $parensEndingSelf = $afterCloseBeforeOpen;
        }
     
        my $rest = substr($parsedText2, $i);
        $rest =~ s/^[^\(]*//;
        $rest =~ /\(+(\S+)\s/;
        $selfCat = "S".$1;
        

        my $posSelf = $i+1;
        
        #find parent and left sib if exists
        my $nOpenParens = 0;
        my $i = $posSelf - 1;
        while (($i >= 0) && ($nOpenParens < 1)) {
          my $c = substr($parsedText2, $i, 1);
          if ($c eq '(') {
            $nOpenParens++;
            #if found a left sibling
            if ($nOpenParens == 0 and $leftSibCat eq "L") {
              substr($parsedText2, $i) =~ /\(+(\S+)\s/;
              $leftSibCat = "L".$1;
            }
          }
          elsif ($c eq ')') {
            $nOpenParens--;
          }
          $i--;
        }
        #found parent
        substr($parsedText2, $i) =~ /\(+(\S+)\s/;
        $parentCat = "P".$1;
        
        ##find right sibling if it exists
        ##find end of selfCategory--need to know number of closing parens and spaces expected
        my $rest = substr($parsedText2, $pos);

        #take off that many right parens
        my $i = 0;
        while ($parensEndingSelf > 0) {
          if (substr($rest, $i, 1) eq ')') {
            $parensEndingSelf--;
          }
          $i++;
        }
        $rest = substr($rest, $i);
        $rest =~ s/^\s+//s;
 
        #right sib exists if next parens is open
        if ($rest =~ /^(\(+)(\S+)\s/) {
          $rightSibCat = "R".$2;
          #check for VP and traces
          $nOpenParens = 1;
          my $i = 1;
          while ($nOpenParens > 0) {
            if (substr($rest, $i, 3) eq "(VP") {
              $rightSibWVP = "RSVPP";
            }
            if (substr($rest, $i, 7) eq "(-NONE-") {
              $rightSibWTrace = "RSTP";
            }
            if (substr($rest, $i, 1) eq "(") {
              $nOpenParens++;
            }
            if (substr($rest, $i, 1) eq ")") {
              $nOpenParens--;
            }
            $i++;
          }
        }
        my $sense = "NonDisc";
        my $outputLine = "";
        $conn =~ s/\s//g; #strip out spaces so one features
        my @indivFeatures = ($conn, $selfCat, $parentCat, $leftSibCat, $rightSibCat, $rightSibWVP);
#my @indivFeatures = ($conn);
        my @features = (); 

        #remove functional tags
        foreach my $i (0..scalar(@indivFeatures)-1) {
          $indivFeatures[$i] =~ s/(.[^-]+)-.*/$1/;
        }

        #all features individually
        foreach my $f (@indivFeatures) {
          #strip out non-alpha characters in features so mallet output will work
          $f =~ s/\$/Dollar/g;
          $f =~ s/\`\`/OpenQuote/g;
          $f =~ s/\'\'/CloseQuote/g;
          $f =~ s/\(/OpenParen/g;
          $f =~ s/\)/CloseParen/g;
          $f =~ s/\,/Comma/g;
          $f =~ s/\-\-/Dash/g;
          $f =~ s/\./EndSent/g;
          $f =~ s/\:/Colon/g;
          $f =~ s/[^A-Za-z ]//g;
          push @features, $f;
        } 
        #all interactions
        foreach my $i (0..scalar(@indivFeatures)-2) {
          foreach my $j ($i+1..scalar(@indivFeatures)-1) {
            push @features, $indivFeatures[$i].$indivFeatures[$j];
          }
       }


       ###update newParsedText
       ###find first unannotated connective
       $regex = '\\b'.join('\\)*\s+(\s|\n|\\(\\S*|\))*', @words) . '\\)';
       if ($newParsedText =~ m/$regex/gis) {
##find best label  
         my %labelEvid = ();
         foreach my $l (@posLabels) {
           $labelEvid{$l} += $labelWeights{$l}{"<default>"};
           foreach my $f (@features) {
             $labelEvid{$l} += $labelWeights{$l}{lc($f)};
           }
         }
         my $maxLabel = $posLabels[0];
         my $max = $labelEvid{$maxLabel};
         foreach my $i (0..scalar(@posLabels)-1) {
           if ($labelEvid{$posLabels[$i]} > $max) {
             $maxLabel = $posLabels[$i];
             $max = $labelEvid{$maxLabel};
           }
        }
#find where to place it in parse file
         my $pos = pos($newParsedText);
         my $beginPos = $pos - length($&);
         my $insertString = $&;
         foreach my $w (@words) {
           $insertString =~ s/(\b$w)\)/$1#$idInFile#$maxLabel\)/i;
         }
         $newParsedText = substr($newParsedText, 0, $beginPos) . $insertString . substr($newParsedText, $pos);
         #if long-distance, check for second part before end of sentence
         if ($longDistConnectives{lc($conn)}) {
           my $secondHalf = $longDistConnectives{lc($conn)};
           my $secondPos = -1;
           my $rest =  substr($newParsedText, $pos+length($insertString));
           if ($rest =~ m/\b$secondHalf[^\)]*\b/g) {
             my $secondPos = pos($rest);
             my $len = length($&);
             if (not substr($rest,0,$secondPos) =~ /\(\./) {
               $newParsedText = substr($newParsedText, 0, $pos+length($insertString)+$secondPos-$len) . $secondHalf."#".$idInFile."#".$maxLabel.substr($newParsedText, $pos+length($insertString)+$secondPos);
             }
           }
         }         
         $overallId++;
         $idInFile++;
       }
     }
 
   }   
   if (-d $outputDir) {
     open(FOUT, ">$outputDir/$ending.disc");
     print FOUT $newParsedText, "\n";
     close(FOUT);
     print STDERR "Wrote $outputDir$ending.disc\n";
   }
   elsif ($outputDir) {
     open(FOUT, ">$outputDir");
     print FOUT $newParsedText, "\n";
     close(FOUT);
     print STDERR "Wrote $outputDir\n";
   }
   else {
     print $newParsedText, "\n";
   }
  }
