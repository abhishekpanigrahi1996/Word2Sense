#define STRIP_FLAG_HELP 0

#include <gflags/gflags.h>

#include "Bigraph.hpp"
#include "Vocab.hpp"
#include "Utils.hpp"
#include <exception>
#include <algorithm>
using namespace std;

DEFINE_string(prefix, "./prefix", "prefix of output files");
DEFINE_string(vocab_in, "", "input vocabulary file");
DEFINE_string(vocab_out, "", "output vocabulary file");
DEFINE_string(input, "", "input file");
DEFINE_string(output, "", "output file");
DEFINE_string(type, "text", "type of input: text, uci, libsvm");
DEFINE_int32(skip, 2, "skip num of words at first of each line (only for text)");
DEFINE_bool(test, false, "test mode (throw away unseen words)");

template <bool testMode = false>
void parse_document(string &line, std::vector<TVID> &v, Vocab &vocab)
{
    std::istringstream sin(line);
    std::string doc_id, word_id;
    v.clear();
    if (FLAGS_type == "text")
    {
    }
    else if (FLAGS_type == "uci")
    {
    }
    else if (FLAGS_type == "libsvm")
    {
    }
    else
    {
        throw std::runtime_error(std::string("Unknown input type " + FLAGS_type));
    }
}

template <bool testMode = false>
void text_to_bin(std::string in, std::string out)
{
    size_t num_tokens = 0;
    Vocab v;
    v.load(FLAGS_vocab_in);
    int doc_id = 0;
    std::vector<std::tuple<TVID, TVID, float>> edge_list;
    std::vector<TVID> vlist;
    bool success = ForEachLinesInFile(in, [&](std::string line)
    {
        parse_document<testMode>(line, vlist, v);
	std::istringstream sin(line);
	int doc_id_t, word_id_t;
        float count_id;
	sin >> doc_id_t >> word_id_t >> count_id;
        edge_list.emplace_back(doc_id_t-1, word_id_t-1, count_id);
        doc_id++;
        num_tokens += vlist.size();
    });
    if (!success)
        throw std::runtime_error(std::string("Failed to input file ") + in);
    // Shuffle tokens
    v.store(FLAGS_vocab_out);
    
    if (!testMode)
        Bigraph::Generate(out, edge_list);
    else
        Bigraph::Generate(out, edge_list, v.nWords());
    cout << "Done. Processed " << num_tokens << " tokens." << endl;
}

int main(int argc, char** argv)
{
    gflags::SetUsageMessage("Usage : ./transform [ flags... ]");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (FLAGS_input.empty())
		FLAGS_input = FLAGS_prefix + ".txt";
	if (FLAGS_output.empty())
		FLAGS_output = FLAGS_prefix + ".bin";
	if (FLAGS_vocab_out.empty())
		FLAGS_vocab_out = FLAGS_prefix + ".vocab";
    if (FLAGS_vocab_in.empty() && (FLAGS_test || FLAGS_type != "text"))
        throw runtime_error("Input vocabulary is not specified.");
    if (FLAGS_vocab_in == FLAGS_vocab_out)
        throw runtime_error("Input prefix and output prefix are the same.");

    cout << "Reading corpus from " << FLAGS_input << endl;
    if (FLAGS_test || FLAGS_type != "text")
        cout << "Reading vocabulary from " << FLAGS_vocab_in << endl;
    else
        cout << "Vocabulary will be wrote to " << FLAGS_vocab_out << endl;
    cout << "Output will be wrote as " << FLAGS_output << endl;

    if (FLAGS_test)
        text_to_bin<true>(FLAGS_input, FLAGS_output);
    else 
        text_to_bin<false>(FLAGS_input, FLAGS_output);

	return 0;
}
